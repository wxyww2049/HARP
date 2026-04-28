import re
import sys

def generate_cpuset_rankfile(node_str, max_nodes=None, output_file="rankfile_cpuset.txt"):
    # --- 1. 基础配置 ---
    RANKS_PER_NODE = 16
    TOTAL_CORES = 608
    CORES_PER_RANK = TOTAL_CORES // RANKS_PER_NODE  # 152 / 4 = 38个核
    
    # 简单的 Socket 判定阈值 (假设前76个核在Socket 0，后76个在Socket 1)
    SOCKET_BOUNDARY = 304 

    # --- 2. 解析节点列表 ---
    expanded_nodes = []
    with open(node_str, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                expanded_nodes.append(parts[0])
                
    # === 新增：对节点列表进行自然排序 ===
    # 使用正则表达式提取字符串中的数字，确保类似 node2 排在 node10 前面
    expanded_nodes.sort(key=lambda x: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', x)])

    # 截取所需的最大节点数 (在排序之后截取，保证每次取出的节点是最靠前的 N 个)
    if max_nodes is not None:
        expanded_nodes = expanded_nodes[:max_nodes]
        
    # print(f"generate rankfile for ###{len(expanded_nodes)}###")
    
    # --- 3. 生成 Rankfile ---
    lines = []
    global_rank = 0

    for node in expanded_nodes:
        # 在当前节点生成 RANKS_PER_NODE 个 rank
        for i in range(RANKS_PER_NODE):
            # 计算当前 Rank 占用的核心范围
            start_core = i * CORES_PER_RANK
            end_core = start_core + CORES_PER_RANK - 1
            
            # 判断属于哪个 Socket
            socket_id = 0 if start_core < SOCKET_BOUNDARY else 1
            start_core = start_core % SOCKET_BOUNDARY 
            end_core = end_core % SOCKET_BOUNDARY 

            # 格式化输出
            # 注意：这里使用全局 Rank ID 和 全局 Core ID
            line = f"rank {global_rank}={node} slot={socket_id}:{start_core}-{end_core}"
            lines.append(line)
            global_rank += 1

    # 打印到控制台
    for l in lines:
        print(l)
        
    # 可选：如果需要写入到文件，可以取消下面这两行的注释
    # with open(output_file, 'w', encoding='utf-8') as out_f:
    #     out_f.write('\n'.join(lines) + '\n')

# --- 输入数据 ---
if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python gen_rankfile.py <node_list.txt> [n]")
        sys.exit(1)

    node_list_input = sys.argv[1]
    max_nodes = None
    
    if len(sys.argv) == 3:
        max_nodes = int(sys.argv[2])
        if max_nodes <= 0:
            raise ValueError("n must be a positive integer")

    generate_cpuset_rankfile(node_list_input, max_nodes=max_nodes)