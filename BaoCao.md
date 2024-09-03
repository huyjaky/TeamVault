# 1.Khảo sát các cách tiếp cận

-  **Cách 1:** Hệ thống invariant and specific modality (ISM) sử dụng mạng nơ-ron đồ thị (Graph Neural Networks - GNN) với hai không gian đặc trưng khác nhau: một không gian đại diện chung cho cả hai loại dữ liệu (modality-invariant) và một không gian riêng cho từng loại dữ liệu (modality-specific).

-  **Cách 2:** sử dụng model UniformerV2+MEAA để giải quyết bài toán xác định hình ảnh bạo lực kết hợp với bộ dữ liệu đầu vào được kiểm soát từ indentify data cho khả năng xử lý hình ảnh tốt hơn bằng cách thay Q bằng q learning.

-  **Cách 3:** BLIP-2, một chiến lược huấn luyện trước chung và hiệu quả, tận dụng các bộ mã hóa hình ảnh đã được huấn luyện sẵn và các mô hình ngôn ngữ lớn đã được đóng băng. BLIP-2 kết nối khoảng cách giữa các mô hình bằng một Transformer Querying nhẹ, được huấn luyện trước trong hai giai đoạn. Giai đoạn đầu tiên là học biểu diễn giữa thị giác và ngôn ngữ từ một bộ mã hóa hình ảnh đã đóng băng. Giai đoạn thứ hai là học sinh ngôn ngữ từ hình ảnh sang ngôn ngữ từ một mô hình ngôn ngữ đã đóng băng.
# 2. Giải pháp được chọn
-  **Giải pháp:** sử dụng mô hình BLIP-2 kết hợp với mô hình ISM để cho ra khả năng xử lý ảnh cải thiện đáng kể so với model xử lý hình ảnh cũ của ISM, BLIP-2

-  **Lý do chọn:** 
	-  Giải pháp này cung cấp sự kết hợp giữa modality-invariant (giảm khoảng cách giữa các phương thức) và modality-specific (nắm bắt các đặc điểm riêng biệt của từng phương thức). Điều này giúp tạo ra một cái nhìn toàn diện về meme và tăng cường khả năng phát hiện meme độc hại so với các mô hình khác. 
	-  Khả năng nhận diện mối liên hệ giữa hình ảnh và chữ của BLIP-2
	-  Thử nghiệm trên dữ liệu công khai cho thấy khung ISM, BLIP-2 đề xuất cải thiện đáng kể so với các phương pháp hiện có, đồng thời đạt hiệu suất cạnh tranh trong việc phát hiện meme độc hại.

- ***Sơ đồ:*** 
![Overview map](./Images/Screenshot%202024-09-02%20205647.png)


# 3.Thu thập & Phân tích dữ liệu & Tiền xử lý dữ liệu & Tăng cường dữ liệu
- **Thu thập dữ liệu:** 
	- Sử dụng các bộ dữ liệu meme từ nhiều nguồn khác nhau như các cuộc thi và các bài viết liên quan đến các chủ đề cụ thể như chính trị, COVID-19, v.v. 
	- Cào dữ liệu về meme trên google

- **Phân tích và tiền xử lý dữ liệu:** Dữ liệu được phân tích để xác định các tính năng quan trọng, sau đó được chuẩn bị cho mô hình hóa bằng cách loại bỏ nhiễu và tiêu chuẩn hóa.

- **Tăng cường dữ liệu:** Sử dụng các kỹ thuật như augmentation dữ liệu và học chuyển giao để cải thiện độ chính xác của mô hình.

- **Minh họa:**

![Pasted image 20240902220150](./Images/Pasted%20image%2020240902220150.png)



# 4.Lựa chọn và Huấn luyện mô hình

- **Mô hình chọn lựa:** ISM dual-stream (CLIP, ALBEF, BLIP).

- **Lý do chọn:** Các mô hình này cung cấp khả năng phân tích sâu và tương tác giữa các phương thức, tạo điều kiện thuận lợi cho việc phát hiện meme độc hại.

- **Kiến trúc mô hình:** 

![Pasted image 20240902212024](./Images/Pasted%20image%2020240902212024.png)


![Pasted image 20240902084316](./Images/Pasted%20image%2020240902084316.png)


- **Tham số huấn luyện:** 
	-  **Learning Rate:** Tốc độ học xác định kích thước của các bước mà mô hình thực hiện để cập nhật trọng số trong mỗi lần lặp. Tốc độ học quá cao mô hình không hội tụ hoặc hội tụ không ổn định, trong khi tốc độ học quá thấp làm quá trình huấn luyện chậm và không đạt được tối ưu.
	
	-  **Batch Size:** số lượng mẫu được sử dụng trong mỗi bước cập nhật trọng số. Batch Size nhỏ có thể giúp mô hình học nhanh hơn và sử dụng bộ nhớ hiệu quả hơn, nhưng có thể gây ra biến động lớn trong gradient. Batch Size lớn có thể cải thiện độ chính xác của mô hình nhưng yêu cầu nhiều bộ nhớ hơn.
	
	-  **Number of Epochs:** số lần toàn bộ tập dữ liệu huấn luyện được đưa qua mô hình. Tăng số lượng epochs có thể giúp mô hình học tốt hơn, nhưng có nguy cơ dẫn đến  overfitting.

	-  **Optimizer:**  thuật toán được sử dụng để cập nhật trọng số của mô hình dựa trên gradient của hàm mất mát. Các optimizer phổ biến bao gồm Gradient Descent, Adam, RMSprop, và SGD (Stochastic Gradient Descent).
	
	-  **Loss Function:** Hàm mất mát đo lường sự khác biệt giữa dự đoán của mô hình và giá trị thực. Việc lựa chọn hàm mất mát phù hợp là rất quan trọng, vì nó ảnh hưởng đến cách mô hình học và cải thiện.
	
	-  **Activation Functions:** Hàm kích hoạt quyết định đầu ra của mỗi neuron trong mạng nơ-ron. Các hàm kích hoạt phổ biến bao gồm ReLU (Rectified Linear Unit), sigmoid, và tanh. Chúng giúp mô hình học các mối quan hệ phi tuyến trong dữ liệu.

# 5.Đánh giá

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

# 6.Triển khai

- **Phát triển mô hình nhận diện ảnh độc hại**

- **Phát triển Extension:**
	- Lựa chọn nền tảng: trình duyệt web, app
    - Tạo một module tích hợp mô hình nhận diện ảnh.
	- Module này sẽ tải mô hình đã huấn luyện và sử dụng nó để phân tích hình ảnh mà người dùng tải lên hoặc truy cập.
	
- **Triển khai mô hình và extension**
	- Chuẩn bị môi trường triển khai: server nội bộ.
	- Triển khai mô hình: Đưa mô hình lên server. Đảm bảo mô hình sẵn sàng nhận và xử lý yêu cầu nhận diện ảnh.
	- Triển khai extension: Đảm bảo extension có thể kết nối với mô hình đã triển khai và xử lý các hình ảnh người dùng tải lên hoặc truy cập.

# 7.Mã nguồn giải pháp

- **Mã nguồn ISM:** https://github.com/caskcsg/ISM
- **Mã nguồn BLIP-2:** https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip_2/modeling_blip_2.py?fbclid=IwZXh0bgNhZW0CMTAAAR1JCGGaRktFc0RMWsFTlbzHQH2aXdnF8wyuQdz3NfBxRNYiT5_XawkjjgI_aem_6eK59qx5RUshMPG5YJ2szw
- **Dữ liệu từ Hateful Memes Challenge:** https://www.kaggle.com/datasets/williamberrios/hateful-memes
