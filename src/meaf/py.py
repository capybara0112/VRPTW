import matplotlib.pyplot as plt
import pandas as pd
import os

# 1. Xác định đường dẫn file CSV (tự động tìm từ thư mục chứa file script này)
# Giả sử cấu trúc: java/src/meaf/py.py -> lùi 2 cấp để ra java/ rồi vào results/
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "..", "..", "results", "C101_convergence.csv")

# Kiểm tra nếu file không tồn tại ở đường dẫn trên, thử tìm ở thư mục hiện tại
if not os.path.exists(csv_path):
    csv_path = os.path.join("results", "C101_convergence.csv")

print(f"[*] Đang đọc dữ liệu từ: {csv_path}")

try:
    # 2. Đọc dữ liệu từ file CSV
    df = pd.read_csv(csv_path)

    # 3. Cấu hình biểu đồ
    plt.figure(figsize=(12, 7))
    
    # Vẽ các đường thuật toán
    plt.plot(df['Generation'], df['GA_Cost'], label='GA (Genetic Algorithm)', color='orange', alpha=0.7)
    plt.plot(df['Generation'], df['PSO_Cost'], label='PSO (Particle Swarm)', color='green', alpha=0.7)
    plt.plot(df['Generation'], df['ACS_Cost'], label='ACS (Ant Colony)', color='blue', alpha=0.7)
    
    # Đường MEAF vẽ đậm hơn để làm nổi bật (kết quả của framework)
    plt.plot(df['Generation'], df['MEAF_Cost'], label='MEAF (Proposed Framework)', color='red', linewidth=3)

    # 4. Thêm chi tiết cho biểu đồ giống bài báo
    plt.xlabel('Generations', fontsize=12)
    plt.ylabel('Total Distance / Cost', fontsize=12)
    plt.title('Convergence Curves Comparison (Dataset: C101)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    # 5. LƯU THÀNH FILE ẢNH
    output_image = os.path.join(os.path.dirname(csv_path), "convergence_chart.png")
    plt.savefig(output_image, dpi=300) # dpi=300 để ảnh nét như in báo
    
    print(f"[✔] Đã lưu biểu đồ thành công tại: {output_image}")
    
    # Hiển thị lên màn hình
    plt.show()

except Exception as e:
    print(f"[X] Lỗi: {e}")