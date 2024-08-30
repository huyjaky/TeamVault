# 1.Khảo sát các cách tiếp cận

- **Cách 1:** Hệ thống invariant and specific modality (ISM) sử dụng mạng nơ-ron đồ thị (Graph Neural Networks - GNN) với hai không gian đặc trưng khác nhau: một không gian đại diện chung cho cả hai loại dữ liệu (modality-invariant) và một không gian riêng cho từng loại dữ liệu (modality-specific).

- **Cách 2:** sử dụng model UniformerV2+MEAA để giải quyết bài toán xác định hình ảnh bạo lực kết hợp với bộ dữ liệu đầu vào được kiểm soát từ indentify data cho khả năng xử lý hình ảnh tốt hơn bằng cách thay Q bằng q learning 

# 2. Giải pháp được chọn
- **Giải pháp:** sử dụng mô hình UniformerV2+MEAA kết hợp với mô hình ISM để cho ra khả năng xử lý ảnh cải thiện đáng kể so với model xử lý hình ảnh cũ của ISM

- **Lý do chọn:** 
	- Giải pháp này cung cấp sự kết hợp giữa modality-invariant (giảm khoảng cách giữa các phương thức) và modality-specific (nắm bắt các đặc điểm riêng biệt của từng phương thức). Điều này giúp tạo ra một cái nhìn toàn diện về meme và tăng cường khả năng phát hiện meme độc hại so với các mô hình khác. 
	- Khả năng nhận diện ảnh độc hại của UniformerV2+MEA cao
	- Thử nghiệm trên năm bộ dữ liệu công khai cho thấy khung ISM đề xuất cải thiện đáng kể so với các phương pháp hiện có, đồng thời đạt hiệu suất cạnh tranh trong việc phát hiện meme độc hại.

- ***Sơ đồ:*** ==***HUY NHỚ LÀM***==
# 2.Thu thập & Phân tích dữ liệu & Tiền xử lý dữ liệu & Tăng cường dữ liệu
- **Thu thập dữ liệu:** 
	- Sử dụng các bộ dữ liệu meme từ nhiều nguồn khác nhau như các cuộc thi và các bài viết liên quan đến các chủ đề cụ thể như chính trị, COVID-19, v.v. 
	- Cào dữ liệu về meme trên google

- **Phân tích và tiền xử lý dữ liệu:** Dữ liệu được phân tích để xác định các tính năng quan trọng, sau đó được chuẩn bị cho mô hình hóa bằng cách loại bỏ nhiễu và tiêu chuẩn hóa.

- **Tăng cường dữ liệu:** Sử dụng các kỹ thuật như augmentation dữ liệu và học chuyển giao để cải thiện độ chính xác của mô hình.

- **Minh họa:** ==***HUY NHỚ LÀM***==

# 3.Lựa chọn và Huấn luyện mô hình

- **Mô hình chọn lựa:** ISM dual-stream (CLIP, ALBEF, BLIP) và UniformerV2+MEAA.

- **Lý do chọn:** Các mô hình này cung cấp khả năng phân tích sâu và tương tác giữa các phương thức, tạo điều kiện thuận lợi cho việc phát hiện meme độc hại.

- **Kiến trúc mô hình:** ==***HUY NHỚ LÀM***==

- **Tham số huấn luyện:** 
	-  **Learning Rate:** Tốc độ học xác định kích thước của các bước mà mô hình thực hiện để cập nhật trọng số trong mỗi lần lặp. Tốc độ học quá cao mô hình không hội tụ hoặc hội tụ không ổn định, trong khi tốc độ học quá thấp làm quá trình huấn luyện chậm và không đạt được tối ưu.
	
	- **Batch Size:** số lượng mẫu được sử dụng trong mỗi bước cập nhật trọng số. Batch Size nhỏ có thể giúp mô hình học nhanh hơn và sử dụng bộ nhớ hiệu quả hơn, nhưng có thể gây ra biến động lớn trong gradient. Batch Size lớn có thể cải thiện độ chính xác của mô hình nhưng yêu cầu nhiều bộ nhớ hơn.
	
	- **Number of Epochs:** số lần toàn bộ tập dữ liệu huấn luyện được đưa qua mô hình. Tăng số lượng epochs có thể giúp mô hình học tốt hơn, nhưng có nguy cơ dẫn đến  overfitting.

	- **Optimizer:**  thuật toán được sử dụng để cập nhật trọng số của mô hình dựa trên gradient của hàm mất mát. Các optimizer phổ biến bao gồm Gradient Descent, Adam, RMSprop, và SGD (Stochastic Gradient Descent).
	
	- **Loss Function:** Hàm mất mát đo lường sự khác biệt giữa dự đoán của mô hình và giá trị thực. Việc lựa chọn hàm mất mát phù hợp là rất quan trọng, vì nó ảnh hưởng đến cách mô hình học và cải thiện.
	
	- **Regularization:** là các kỹ thuật để ngăn ngừa mô hình overfitting bằng cách thêm các điều kiện bổ sung vào hàm mất mát. Các phương pháp phổ biến bao gồm L1 và L2 regularization, dropout, và data augmentation.
	
	- **Activation Functions:** Hàm kích hoạt quyết định đầu ra của mỗi neuron trong mạng nơ-ron. Các hàm kích hoạt phổ biến bao gồm ReLU (Rectified Linear Unit), sigmoid, và tanh. Chúng giúp mô hình học các mối quan hệ phi tuyến trong dữ liệu.
	
	- **Momentum:** là một kỹ thuật để tăng tốc độ hội tụ và giảm thiểu biến động trong quá trình huấn luyện bằng cách tích lũy gradient qua các bước cập nhật.
	
	- **Weight Initialization:** Khởi tạo trọng số ảnh hưởng đến cách mà các trọng số của mô hình được bắt đầu trước khi huấn luyện. Việc khởi tạo trọng số tốt có thể giúp mô hình hội tụ nhanh hơn và hiệu quả hơn.
	
	- **Learning Rate Scheduler:** Lịch trình tốc độ học điều chỉnh tốc độ học trong suốt quá trình huấn luyện, giúp mô hình học nhanh ở giai đoạn đầu và tinh chỉnh khi gần đạt được tối ưu.
	
	- **Dropout Rate:** Là tỷ lệ phần trăm của các neuron bị tắt ngẫu nhiên trong mỗi lần lặp huấn luyện để ngăn ngừa overfitting và tăng khả năng tổng quát của mô hình.

# 4.Đánh giá

- **Xây dựng bộ dữ liệu:** 
	- **Train set (Tập huấn luyện)**: 
		- 60-80% dữ liệu
		- **Mục đích**: Được sử dụng để huấn luyện mô hình. Mô hình học từ dữ liệu này, điều chỉnh trọng số dựa trên loss function.
	- **Validation set (Tập kiểm tra)**: 
		- 10-20% dữ liệu
	    - **Mục đích**: Được sử dụng để điều chỉnh siêu tham số (hyperparameters) và theo dõi hiệu suất của mô hình trong quá trình huấn luyện. Tập này giúp ngăn ngừa overfitting bằng cách cung cấp một tập dữ liệu mà mô hình chưa thấy trong quá trình huấn luyện.
	- **Test set (Tập kiểm thử)**: 
		- 10-20% dữ liệu
	    - **Mục đích**: Được sử dụng để đánh giá hiệu suất cuối cùng của mô hình sau khi đã hoàn tất huấn luyện và điều chỉnh siêu tham số. Tập này chỉ được sử dụng một lần sau khi mô hình đã sẵn sàng để đảm bảo rằng kết quả không bị thiên lệch.

- Cách chia bộ dữ liệu: Random Split, Stratified Split, Time-based Split

- **Chỉ số đánh giá:** 
	- **Accuracy:** Tỷ lệ phần trăm các mẫu được dự đoán chính xác so với tổng số mẫu. Đây là một chỉ số tổng quát để đánh giá hiệu suất mô hình.
	    
	- **Precision:** Tỷ lệ phần trăm các mẫu dự đoán là dương tính (harmful) mà thực sự là dương tính. Nó giúp đánh giá mức độ mà mô hình không đưa ra dự đoán sai dương tính.
	    
	- **Recall:** Tỷ lệ phần trăm các mẫu dương tính thực sự được mô hình phát hiện ra. Đây là một chỉ số quan trọng khi muốn đảm bảo rằng tất cả các trường hợp dương tính đều được nhận diện.
	    
	- **F1-Score:** Trung bình điều hòa của Precision và Recall, đặc biệt hữu ích khi cần cân bằng giữa hai chỉ số này.
	    
	- **Confusion Matrix:** Một công cụ trực quan hóa kết quả phân loại, cho thấy số lượng dự đoán đúng và sai trong các lớp dương tính và âm tính.

# 5.Triển khai

- **Thiết kế kiến trúc hệ thống:** Tích hợp mô hình vào một hệ thống API, xây dựng giao diện người dùng (UI) đơn giản để tương tác với mô hình. Sử dụng các công cụ và dịch vụ như Flask hoặc FastAPI cho API, Docker để container hóa và deploy trên các nền tảng cloud như AWS hoặc GCP.

- **Hosting:** Đưa hệ thống lên website thông qua các dịch vụ hosting phù hợp, tối ưu hóa để đảm bảo hiệu suất và khả năng mở rộng.

# 6.Mã nguồn giải pháp

- **Mã nguồn ISM:** https://github.com/caskcsg/ISM
- **Mã nguồn CUE-net:** https://github.com/huyjaky/TeamVault/tree/modal_cuenet
- **Dữ liệu từ Hateful Memes Challenge:** https://www.kaggle.com/datasets/williamberrios/hateful-memes
