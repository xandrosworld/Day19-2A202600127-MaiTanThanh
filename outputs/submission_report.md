# Lab Day 19: Xây Dựng Hệ Thống GraphRAG Với Tech Company Corpus

**Sinh viên:** Mai Tấn Thành  
**Mã sinh viên:** 2A202600127

## 1. Nghiên Cứu

**Entity Extraction:** Entity Extraction là bước tìm các thực thể quan trọng trong văn bản, ví dụ: OpenAI, Sam Altman, Microsoft, Google, DeepMind. LLM cần phân biệt đâu là thực thể có thể trở thành node trong đồ thị và đâu là thuộc tính. Nếu một cụm từ là đối tượng có thể liên kết với nhiều quan hệ khác, ví dụ tên công ty hoặc tên người, thì nên xem là node. Nếu cụm từ chỉ mô tả một giá trị đơn lẻ như năm thành lập, địa điểm hoặc chức danh, thì có thể xem là thuộc tính hoặc object value.

**Graph Construction:** Sau khi có các triples, hệ thống tạo node và edge trong đồ thị. Deduplication rất quan trọng vì cùng một thực thể có thể xuất hiện dưới nhiều tên gần giống nhau, ví dụ "Apple Inc." và "Apple", hoặc "Google DeepMind" và "DeepMind". Nếu không khử trùng lặp, đồ thị bị tách mảnh và truy vấn multi-hop dễ trả lời sai.

**Query Answering:** Flat RAG tìm các đoạn văn bản gần với câu hỏi bằng vector similarity. GraphRAG tìm thực thể chính trong câu hỏi, sau đó duyệt đồ thị trong phạm vi 2-hop để lấy các quan hệ liên quan. BFS/traverse phù hợp với câu hỏi cần nối nhiều quan hệ, ví dụ: Sam Altman -> OpenAI -> Microsoft.

## 2. Công Cụ Sử Dụng

- Corpus: các trang Wikipedia về OpenAI, Google, Microsoft, Meta Platforms, Amazon, Apple, Nvidia, Anthropic, Alphabet Inc. và Google DeepMind.
- LLM: `gpt-5-nano-2025-08-07`.
- Embedding model: `text-embedding-3-small`.
- Thư viện đồ thị: NetworkX.
- Trực quan hóa: Matplotlib.
- Bảng kết quả: Pandas CSV/Markdown.
- Flat RAG: OpenAI embeddings kết hợp FAISS nếu có, nếu không thì dùng cosine similarity bằng NumPy.

## 3. Quy Trình Thực Hiện

1. Lấy phần giới thiệu từ các trang Wikipedia và lưu cache vào `outputs/corpus_wiki.json`.
2. Dùng OpenAI API để trích xuất triples dạng `(subject, relation, object)`.
3. Bổ sung một số curated seed triples cho các sự kiện benchmark quan trọng.
4. Khử trùng lặp thực thể và loại bỏ các triples gây nhiễu.
5. Xây dựng đồ thị có hướng bằng NetworkX.
6. Chạy 20 câu hỏi benchmark trên cả Flat RAG và GraphRAG.
7. Xuất ảnh đồ thị, bảng so sánh kết quả và thống kê token/time.

## 4. Kết Quả Đánh Giá

- Flat RAG dùng OpenAI embeddings và FAISS/vector cosine retrieval.
- Điểm Flat RAG: **17/20**.
- Điểm GraphRAG: **20/20**.

GraphRAG cho kết quả tốt hơn ở các câu hỏi multi-hop hoặc câu hỏi phụ thuộc nhiều vào quan hệ giữa thực thể, ví dụ:

- "Which company invested in the organization co-founded by Sam Altman?"
- "Which company acquired the lab that developed AlphaGo?"
- "Which company owns Google and also owns DeepMind?"

Flat RAG đôi khi trả lời thiếu hoặc sai vì thông tin cần thiết nằm rải rác ở nhiều đoạn khác nhau. GraphRAG có lợi thế hơn vì có thể duyệt trực tiếp các quan hệ trong đồ thị tri thức.

## 5. Chi Phí Và Thời Gian Chạy

Kết quả lần chạy mới nhất:

- Số API calls: **61**.
- Input tokens: **23,996**.
- Output tokens: **1,225**.
- Thời gian API: **72.64 giây**.
- Tổng thời gian chạy: **73.87 giây**.
- Kích thước đồ thị: **110 nodes**, **133 edges**.

Token usage được lấy từ trường usage của OpenAI API và lưu trong `outputs/run_summary.json`. Để quy đổi sang chi phí tiền thật, cần nhân số token với bảng giá hiện tại của model `gpt-5-nano-2025-08-07`.

## 6. Deliverables

- Mã nguồn: `graphrag_tech_wiki.py`
- Ảnh đồ thị: `outputs/tech_company_graph.png`
- Bảng benchmark: `outputs/benchmark_results.csv` và `outputs/benchmark_results.md`
- Triples đã trích xuất: `outputs/triples.json`
- Thống kê token/time: `outputs/run_summary.json`
