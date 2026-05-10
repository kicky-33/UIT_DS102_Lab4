# LAB 4: DECISION TREE & RANDOM FOREST

## 1. Giới thiệu dự án
Tên: La Gia Hân
MSSV: 24520448

Dự án này thực hiện cài đặt và đánh giá hai thuật toán học máy: **Decision Tree** và **Random Forest** trên tập dữ liệu **Wine Quality**. Mục tiêu cốt lõi là so sánh hiệu suất giữa việc tự hiện thực thuật toán bằng **NumPy** và sử dụng thư viện chuyên dụng **Scikit-learn**.

Link dataset: [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)

## 2. Cấu trúc thư mục

Dự án được tổ chức theo cấu trúc module như sau, nhằm tối ưu hóa việc tái sử dụng mã nguồn:

```text
LAB4_DS102/
├── data/                   # Chứa dataset (winequality-red.csv, white.csv)
├── src/                    # Các module xử lý logic cốt lõi
│   ├── DecisionTree.py     # Cài đặt Decision Tree (Assig 1)
│   ├── DecisionTreeRF.py   # Decision Tree tối ưu cho Random Forest
│   ├── RandomForest.py     # Cài đặt Random Forest (Assig 2)
│   ├── data_prepare.py     # Tiền xử lý dữ liệu & Stratified Sampling
│   ├── GridSearch_a1.py    # Tuning cho Assignment 1
│   └── GridSearch_a2.py    # Tuning cho Assignment 2
├── 1.py                    # Script chạy Assignment 1
├── 2.py                    # Script chạy Assignment 2
├── 3.py                    # Script chạy Assignment 3 (Library)
└── README.md               # Báo cáo tổng hợp

```

## 3. Nội dung thực hiện

### Assignment 1: Decision Tree (NumPy)

* Hiện thực thuật toán **CART** (Classification and Regression Trees) bằng NumPy.
* Sử dụng tiêu chí **Entropy** để tính toán độ tinh khiết và cực đại hóa **Information Gain**.
* Thực hiện **Grid Search** để tìm độ sâu tối ưu nhằm kiểm soát hiện tượng **Overfitting**.

### Assignment 2: Random Forest (NumPy)

* Hiện thực kỹ thuật **Bagging** (Bootstrap Aggregating).
* Mỗi cây được huấn luyện trên một tập mẫu **Bootstrap** và một tập con các đặc trưng ngẫu nhiên (Feature Randomness).
* Kết hợp dự đoán thông qua cơ chế **Majority Voting**.

### Assignment 3: Machine Learning Library

* Sử dụng **Scikit-learn** để huấn luyện mô hình.
* Áp dụng **GridSearchCV** để tối ưu hóa các tham số như `criterion`, `max_depth`, và `n_estimators`.

## 4. Kết quả thực nghiệm

Kết quả dưới đây được ghi nhận sau khi thực hiện **Stratified Sampling** để đảm bảo tính đại diện của dữ liệu:

| Assignment | Mô hình | Tham số tối ưu | F1-Score (Weighted) | RMSE |
| --- | --- | --- | --- | --- |
| **Assig 1** | **DT (NumPy)** | max_depth: 12 | 0.5854 | 0.8775 |
| **Assig 2** | **RF (NumPy)** | n_trees: 30, max_depth: 15 | 0.6553 | 0.6889 |
| **Assig 3** | **DT (Library)** | criterion: entropy, max_depth: 20 | 0.6046 | 0.8385 |
| **Assig 3** | **RF (Library)** | n_estimators: 100, max_depth: 20 | **0.6783** | **0.6510** |

## 5. Phân tích & Nhận xét

### 5.1. So sánh Decision Tree và Random Forest

* Kết quả cho thấy **Random Forest** vượt trội hơn hẳn Decision Tree đơn lẻ trên cả hai phương diện NumPy và Thư viện. F1-score tăng từ **0.58** lên **0.65** (NumPy) và từ **0.60** lên **0.67** (Library).
* Việc kết hợp nhiều cây giúp giảm **Variance** (phương sai) và hạn chế Overfitting, dẫn đến sai số **RMSE** thấp hơn đáng kể.

### 5.2. Hiệu quả của Thư viện chuyên dụng

* Các mô hình từ **Scikit-learn** đạt hiệu suất cao nhất nhờ các kỹ thuật tối ưu hóa thuật toán sâu và cơ chế cắt tỉa cây hiệu quả.
* Thời gian huấn luyện và Tuning bằng thư viện nhanh hơn đáng kể so với việc thực hiện thủ công bằng NumPy.

### 5.3. Tầm quan trọng của Hyperparameter Tuning

* Quá trình Tuning cho thấy mô hình rất nhạy cảm với tham số `max_depth`. Việc tìm ra độ sâu tối ưu là chìa khóa để cân bằng giữa Bias và Variance.
* Đối với Random Forest, việc tăng số lượng cây (`n_trees`) giúp mô hình ổn định hơn và đạt kết quả tốt hơn trên tập kiểm tra.


*Báo cáo được thực hiện cho bài Lab 4 môn DS102 - Học máy thống kê.*
