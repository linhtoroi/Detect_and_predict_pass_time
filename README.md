# Detect_and_predict_pass_time

Bài toán dự báo thời gian tắc đường (thời gian đi hết 1 quãng đường trong tương lai 10s, 20s, 30s, ... 100s)


Bài toán áp dụng 2 khối, đầu tiên là mô hình YOLO đã được huấn luyện để trích xuất dữ liệu (số lượng xe, nhãn - thời gian xe đi hết quãng đường), khối thứ hai là mô hình dự đoán thời gian đi hết quãng đường bao gồm 1 tầng LSTM và 1 tầng Linear.


### +) Sau khi clone code, tải yolov3.weights đã được huấn luyện

cd yolo-coco

wget https://pjreddie.com/media/files/yolov3.weights

### +) Lấy dữ liệu từ link drive chuyển vào folder data/video/input

Video được quay trong 2 thời điểm là 9h00 sáng và 3h00 chiều trong 2 ngày tại đoạn đường Xuân Thủy kéo dài từ số 239 Xuân Thủy (tức tòa nhà IPH xuân thủy) đến ngõ 180 Xuân Thủy, trong video có các phương tiện đang lưu thông trên đường. Dữ liệu bao gồm 4 video, mỗi video dài 15 phút

https://drive.google.com/drive/folders/1ztWXbkfKqvNTu5cC7VP9OCqm_-2ZKCnF?fbclid=IwAR1rjzhGd61u3iVDf5i91CdCYN7MtmiQC49LbL4IGk0rIUNUI8LGADZZZ8I


### Chạy trong file experiment.py

### +) Khởi tạo

lfr = LFramework(input_size=số lượng frame 1 lần forward model, hidden_size= của LSTM và đầu vào mạng Linear, output=10 (số lượng muốn dự đoán - đơn vị 10s (10s,20s,30s,... sau))

lfr.defineVideoPath(data_name=tên file video, file_extension=đuôi file video)


### +) Xử lý dữ liệu

lfr.process_data(data_name=tên file video, file_extension=đuôi file video)


### +) Huấn luyện mô hình

lfr.train(data_name=tên file video đã được xử lý dữ liệu, time_data= thời điểm quay video đó (đơn vị giờ))


### +) Chạy test với đầu vào là video

lfr.load_model()

lfr.defineVideoPath(data_name=tên file video muốn test, file_extension=đuôi file video muốn test)

lfr.run(time_data= thời điểm quay video đó (đơn vị giờ))


