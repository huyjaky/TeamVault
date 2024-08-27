# 1.Khảo sát các cách tiếp cận
- **Giải pháp:** Hệ thống invariant and specific modality (ISM) sử dụng mạng nơ-ron đồ thị (Graph Neural Networks - GNN) với hai không gian đặc trưng khác nhau: một không gian đại diện chung cho cả hai loại dữ liệu (modality-invariant) và một không gian riêng cho từng loại dữ liệu (modality-specific).

- **Lý do chọn:** 
	- Giải pháp này cung cấp sự kết hợp giữa modality-invariant (giảm khoảng cách giữa các phương thức) và modality-specific (nắm bắt các đặc điểm riêng biệt của từng phương thức). Điều này giúp tạo ra một cái nhìn toàn diện về meme và tăng cường khả năng phát hiện meme độc hại so với các mô hình khác. 
	- Thử nghiệm trên năm bộ dữ liệu công khai cho thấy khung ISM đề xuất cải thiện đáng kể so với các phương pháp hiện có, đồng thời đạt hiệu suất cạnh tranh trong việc phát hiện meme độc hại.

- ***Sơ đồ:*** 
# 2.Thu thập & Phân tích dữ liệu & Tiền xử lý dữ liệu & Tăng cường dữ liệu
- **Thu thập dữ liệu:** 
	- Sử dụng các bộ dữ liệu meme từ nhiều nguồn khác nhau như các cuộc thi và các bài viết liên quan đến các chủ đề cụ thể như chính trị, COVID-19, v.v. (https://www.kaggle.com/datasets/williamberrios/hateful-memes)
	- Cào dữ liệu về meme trên google

- **Phân tích và tiền xử lý dữ liệu:** Dữ liệu được phân tích để xác định các tính năng quan trọng, sau đó được chuẩn bị cho mô hình hóa bằng cách loại bỏ nhiễu và tiêu chuẩn hóa.

- **Tăng cường dữ liệu:** Sử dụng các kỹ thuật như augmentation dữ liệu và học chuyển giao để cải thiện độ chính xác của mô hình.

# 3.Lựa chọn và Huấn luyện mô hình

- **Mô hình chọn lựa:** ISM với mô hình dual-stream (CLIP, ALBEF, BLIP) làm mô hình xương sống.

- **Lý do chọn:** Các mô hình này cung cấp khả năng phân tích sâu và tương tác giữa các phương thức, tạo điều kiện thuận lợi cho việc phát hiện meme độc hại.

- **Tham số huấn luyện:** Sử dụng learning rate 1e-5, 3e-5, 2e-5 cho các backbone khác nhau, minibatch size 16, training epoch 20.

# 4.Đánh giá

- **Xây dựng bộ dữ liệu validation và test:** Sử dụng các bộ dữ liệu được chia từ các tập dữ liệu chính thức, đảm bảo đa dạng hóa để kiểm tra mô hình.

- **Chỉ số đánh giá:** Accuracy, AUROC, F1, Precision, Recall, MMAE, tùy theo đặc điểm của từng tập dữ liệu và bài toán cụ thể.

# 5.Triển khai

- **Thiết kế kiến trúc hệ thống:** Tích hợp mô hình vào một hệ thống API, xây dựng giao diện người dùng (UI) đơn giản để tương tác với mô hình. Sử dụng các công cụ và dịch vụ như Flask hoặc FastAPI cho API, Docker để container hóa và deploy trên các nền tảng cloud như AWS hoặc GCP.

- **Hosting:** Đưa hệ thống lên website thông qua các dịch vụ hosting phù hợp, tối ưu hóa để đảm bảo hiệu suất và khả năng mở rộng.

# 6.Mã nguồn giải pháp

- **Mã nguồn giải pháp đã chọn:** https://github.com/caskcsg/ISM
- **Dữ liệu từ Hateful Memes Challenge:** https://www.kaggle.com/datasets/williamberrios/hateful-memes