import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import cupy as cp

#TODO: Rewrite the comments in English and make it easier to read.

# --- CUDA Kernel 定義 (核心加速部分) ---
# 這個 C++ Kernel 會在 GPU 上編譯並執行。
# 每個 Thread 處理一個 Agent，直接在 Register/L1 Cache 中完成所有 Lag 的計算。
labs_energy_kernel_code = r'''
extern "C" __global__
void labs_energy(const signed char* __restrict__ population,
                 float* __restrict__ energies,
                 int n_agents, int n_bits) {

    // 1. 計算當前 Thread 的 ID (對應 Agent Index)
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n_agents) return;

    // 2. 將該 Agent 的序列載入 Local Memory (Registers/L1 Cache)
    // 假設 N_BITS 不會超過 128 (通常 LABS 問題都在 100 以內)
    // 這樣可以避免反覆從 Global Memory 讀取，大幅提升速度
    const int MAX_BITS = 128;
    signed char seq[MAX_BITS];

    int offset = tid * n_bits;

    // 載入並轉換 (0 -> -1, 1 -> +1)
    // 這裡我們不建立巨大的中間矩陣，省下 VRAM 和頻寬
    for (int i = 0; i < n_bits; i++) {
        seq[i] = 2 * population[offset + i] - 1;
    }

    float total_energy = 0.0f;

    // 3. 計算所有 Lag 的自相關 (Autocorrelation)
    // 這是原本 Python loop 做的部分，現在被 "Fuse" 進單一 Kernel
    for (int k = 1; k < n_bits; k++) {
        int correlation = 0;
        for (int i = 0; i < n_bits - k; i++) {
            correlation += seq[i] * seq[i+k];
        }
        total_energy += (float)(correlation * correlation);
    }

    // 4. 寫回結果
    energies[tid] = total_energy;
}
'''

class LabsMTS_GPU_Opt:
    def __init__(self, n_bits, n_agents, max_iter):
        self.n_bits = n_bits
        self.n_agents = n_agents
        self.max_iter = max_iter

        # 編譯 Kernel
        self.energy_kernel = cp.RawKernel(labs_energy_kernel_code, 'labs_energy')

    def calculate_labs_energy_batch(self, population):
        """
        使用 Custom CUDA Kernel 進行極速計算
        """
        n_agents = population.shape[0]
        energies = cp.zeros(n_agents, dtype=cp.float32)

        # 設定 Grid 和 Block 大小
        threads_per_block = 256
        blocks_per_grid = (n_agents + threads_per_block - 1) // threads_per_block

        # 執行 Kernel
        # arguments: (population, energies, n_agents, n_bits)
        self.energy_kernel((blocks_per_grid,), (threads_per_block,),
                           (population, energies, n_agents, self.n_bits))

        return energies

    def get_neighbor_batch(self, population, perturbation_strength=1):
        """
        優化後的鄰居生成
        """
        n_agents, n_bits = population.shape
        neighbor = population.copy()

        rows = cp.arange(n_agents).reshape(-1, 1)

        if perturbation_strength == 1:
            # 隨機翻轉 1 個 bit (極快)
            flip_indices = cp.random.randint(0, n_bits, size=(n_agents, 1))
            neighbor[rows, flip_indices] = 1 - neighbor[rows, flip_indices]

        else:
            # 優化：避免對整個矩陣做 argsort
            # 直接隨機生成 k 個索引。雖然可能有重複 (翻轉兩次=沒翻)，但在隨機搜尋中這是可接受的 "隨機性"
            # 這種方法比 argsort 快非常多，且節省大量記憶體
            flip_indices = cp.random.randint(0, n_bits, size=(n_agents, perturbation_strength))

            # 使用 advanced indexing 進行批量翻轉
            # 注意：若同一個 row 選到同樣的 index 兩次，cupy 的行為是未定義或只翻轉一次
            # 為了簡單與速度，我們接受這種微小的碰撞機率 (Birthday Paradox)
            # 或者，可以用 XOR 方式處理重複索引 (XOR 兩次等於沒變)

            # 這裡用一個簡單的迴圈處理 k 次 (k 很小，通常 < 5)
            # 這比生成巨大的 noise 矩陣並排序要快得多
            for k in range(perturbation_strength):
                idx = flip_indices[:, k]
                neighbor[cp.arange(n_agents), idx] = 1 - neighbor[cp.arange(n_agents), idx]

        return neighbor

    def run(self):
        # 初始化
        print(f"Starting Optimized MTS (Custom CUDA) for LABS (N={self.n_bits}) with {self.n_agents} agents...")

        # 記憶體優化：使用 int8 (C++ kernel expects signed char)
        population = cp.random.randint(2, size=(self.n_agents, self.n_bits), dtype=cp.int8)
        energies = self.calculate_labs_energy_batch(population)

        best_idx = cp.argmin(energies)
        global_best_solution = population[best_idx].copy()
        global_best_energy = energies[best_idx]

        history_best_energy = []
        current_global_best_cpu = float(global_best_energy)
        print(f"Initial Best Energy: {current_global_best_cpu}")

        start_time = time.time()

        # 策略參數
        strength_2 = max(2, int(self.n_bits * 0.05))

        for it in range(self.max_iter):
            # --- MTS Search Strategy ---

            # 1. 產生候選者 (Local Search)
            candidate_1 = self.get_neighbor_batch(population, perturbation_strength=1)
            e1 = self.calculate_labs_energy_batch(candidate_1)

            # 2. 更新進步者
            improved_mask = e1 < energies
            population[improved_mask] = candidate_1[improved_mask]
            energies[improved_mask] = e1[improved_mask]

            # 3. 針對未進步者進行廣域搜尋 (Wide Search)
            # 這裡有一個優化機會：不要對 "全部" 人產生 candidate_2，只對 "未進步" 的人產生
            # 但為了保持 array 形狀整齊 (dense array)，通常對全部產生再 mask 覆蓋在 GPU 上反而比較快
            # 除非 "未進步者" 比例極低 (但在 LABS 中，大部分時間大部分人都卡住，所以比例很高)

            not_improved_mask = ~improved_mask
            if cp.any(not_improved_mask):
                candidate_2 = self.get_neighbor_batch(population, perturbation_strength=strength_2)
                e2 = self.calculate_labs_energy_batch(candidate_2)

                improved_s2_mask = (e2 < energies) & not_improved_mask
                population[improved_s2_mask] = candidate_2[improved_s2_mask]
                energies[improved_s2_mask] = e2[improved_s2_mask]

            # 4. 更新全域最佳解
            # 為了減少 Device-Host 同步，每 N 代檢查一次，或者只在 GPU 內做 reduce
            current_best_idx = cp.argmin(energies)
            current_min_energy = energies[current_best_idx]

            # 這裡會觸發一次隱式的同步 (Scalar copy)，但為了監控是必要的
            if current_min_energy < global_best_energy:
                global_best_energy = current_min_energy
                global_best_solution = population[current_best_idx].copy()
                current_global_best_cpu = float(global_best_energy)
                print(f"Iter {it}: New Best: {current_global_best_cpu}")

            history_best_energy.append(current_global_best_cpu)

        cp.cuda.Stream.null.synchronize()
        total_time = time.time() - start_time
        print(f"Optimization finished in {total_time:.2f} seconds. ({(self.max_iter/total_time):.1f} it/s)")

        return (cp.asnumpy(global_best_solution),
                float(global_best_energy),
                cp.asnumpy(energies),
                history_best_energy)

# Visualization function remains the same...
# Main execution block...
if __name__ == "__main__":
    N_BITS = 38
    N_AGENTS = 10000000  # 1000萬 Agent
    MAX_ITER = 100

    mts_opt = LabsMTS_GPU_Opt(N_BITS, N_AGENTS, MAX_ITER)
    best_sol, best_energy, final_energies, history = mts_opt.run()

    # Simple output
    print(f"Final Best Energy: {best_energy}")
