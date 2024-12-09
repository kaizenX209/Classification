# Classification

## Setup

1. Tạo môi trường

- conda create -n skin_classification python=3.11
- Activate môi trường
- conda activate skin_classification

2. Cài đặt các thư viện

```
pip install torch torchvision torchaudio transformers
pip install fastapi python-multipart uvicorn
```

3. Chạy file model.py để test

```
python model.py
```

4. Chạy file app.py để trên api:

```
python app.py
```

- Lên trình duyệt và truy cập địa chỉ sau:

  - http://localhost:8000/docs

- Chọn phương thức POST và upload ảnh để test
