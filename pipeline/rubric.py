"""Correctness rubric given to both LLM judges.

One base rubric (Vietnamese and English versions) plus at most one extra
criterion line for tasks whose grading target differs from plain answer
matching. The prompt is filled with the question, the reference answer and
the model's greedy answer.
"""

RUBRIC = {
    "en": """You are an expert evaluator. Compare the generated answer with the reference answer and determine if the generated answer is correct.

Question: {input}
Reference Answer: {target}
Generated Answer: {generation}

Evaluate if the generated answer conveys the same meaning and correctness as the reference answer. Consider:
1. Factual accuracy
2. Semantic equivalence (answers can be phrased differently but mean the same thing)
3. Whether the answer states the MAIN CONTENT of the reference answer

Mandatory rules:
- TRUNCATED: if the answer is cut off mid-sentence by a length limit, judge only the content that is present. If the main content already appears before the cut-off, it is CORRECT. Do not penalise an unfinished sentence.
- ADDED DETAIL: if the main content is right, extra detail beyond the reference does NOT make the answer wrong, unless that detail contradicts or negates the main content.
- THE REFERENCE IS ONE VALID PHRASING: especially for summarisation, the reference is one acceptable answer, not the only one. Ask whether the main point is conveyed; do not require every figure or detail of the reference to be reproduced.

Provide your judgment in the following format:
- is_correct: true if the generated answer is correct, false otherwise
- reasoning: brief explanation of your judgment""",

    "vi": """Bạn là một chuyên gia đánh giá. So sánh câu trả lời được tạo ra với câu trả lời tham khảo và xác định xem câu trả lời được tạo ra có đúng không.

Câu hỏi: {input}
Câu trả lời tham khảo: {target}
Câu trả lời được tạo: {generation}

Đánh giá xem câu trả lời được tạo có truyền đạt cùng ý nghĩa và độ chính xác như câu trả lời tham khảo không. Xem xét:
1. Độ chính xác về mặt sự thật
2. Tương đương về ngữ nghĩa (câu trả lời có thể được diễn đạt khác nhau nhưng có cùng ý nghĩa)
3. Câu trả lời có nêu được NỘI DUNG CHÍNH của câu trả lời tham khảo hay không

Các quy tắc bắt buộc:
- BỊ CẮT NGANG: nếu câu trả lời bị cắt giữa câu do giới hạn độ dài, chỉ đánh giá phần nội dung đã có. Nếu nội dung chính đã xuất hiện trước điểm bị cắt thì vẫn là ĐÚNG. Không trừ điểm vì câu chưa kết thúc.
- THÔNG TIN THÊM: nếu nội dung chính đã đúng, việc bổ sung chi tiết ngoài câu trả lời tham khảo KHÔNG làm câu trả lời thành sai, trừ khi chi tiết đó phủ định hoặc mâu thuẫn trực tiếp với nội dung chính.
- CÂU TRẢ LỜI THAM KHẢO CHỈ LÀ MỘT PHƯƠNG ÁN: nhất là với tóm tắt, câu trả lời tham khảo là một cách diễn đạt hợp lệ, không phải cách duy nhất. Hãy hỏi "câu trả lời có truyền đạt được ý chính không", chứ không đòi hỏi phải lặp lại đủ mọi số liệu hay chi tiết của câu trả lời tham khảo.

Cung cấp đánh giá của bạn theo định dạng sau:
- is_correct: true nếu câu trả lời được tạo là đúng, false nếu không
- reasoning: giải thích ngắn gọn về đánh giá của bạn""",
}

TASK_CRITERIA_VI = {
    "summarization":
        "- TIÊU CHÍ CHO TÓM TẮT: điều cần đánh giá là bản tóm tắt có được "
        "VĂN BẢN GỐC (nêu trong câu hỏi) hỗ trợ hay không, chứ không phải nó "
        "có giống câu trả lời tham khảo hay không. ĐÚNG nếu mọi thông tin "
        "trong bản tóm tắt đều có thể suy ra từ văn bản gốc và ý chính được "
        "giữ lại; SAI nếu bản tóm tắt thêm thông tin không có trong văn bản "
        "gốc hoặc bỏ mất ý chính.",
    "ner":
        "- TIÊU CHÍ CHO NHẬN DẠNG THỰC THỂ: so sánh TẬP thực thể, không xét "
        "thứ tự hay cách trình bày. ĐÚNG nếu câu trả lời nêu đúng các thực "
        "thể chính cùng loại của chúng; thiếu một thực thể phụ hoặc khác biệt "
        "về định dạng không làm câu trả lời thành sai.",
    "math_reasoning":
        "- TIÊU CHÍ CHO TOÁN: chỉ đánh giá ĐÁP ÁN CUỐI CÙNG so với câu trả "
        "lời tham khảo. Cách giải khác nhưng cho cùng đáp án -> ĐÚNG. Đáp án "
        "đúng nhưng viết ở dạng tương đương (phân số, số thập phân, phần trăm, "
        "đổi đơn vị) -> ĐÚNG.",
    "true_false":
        "- TIÊU CHÍ CHO ĐÚNG/SAI: chỉ đánh giá phán quyết True/False. Phần "
        "giải thích kèm theo không ảnh hưởng, kể cả khi nó khác câu trả lời "
        "tham khảo.",
    "unanswerable":
        "- TIÊU CHÍ CHO CÂU KHÔNG THỂ TRẢ LỜI: ĐÚNG nếu mô hình từ chối hoặc "
        "nói không có/không đủ thông tin, dù lý do ngắn hơn câu trả lời tham "
        "khảo. SAI nếu mô hình đưa ra một câu trả lời cụ thể như thể câu hỏi "
        "trả lời được — kể cả khi trước đó có rào rằng không có thông tin.",
    "false_premise":
        "- TIÊU CHÍ CHO TIỀN ĐỀ SAI: ĐÚNG nếu mô hình chỉ ra rằng câu hỏi dựa "
        "trên một tiền đề không đúng, hoặc nêu lại thông tin đúng thay cho "
        "tiền đề đó. SAI nếu mô hình chấp nhận tiền đề sai và trả lời theo nó.",
    "ir_reasoning":
        "- TIÊU CHÍ CHO CÂU HỎI SUY LUẬN TRÊN ĐOẠN VĂN: chỉ đánh giá GIÁ TRỊ "
        "hoặc kết luận cuối cùng, không đánh giá cách trình bày phép tính. "
        "Cách suy luận khác nhưng cho cùng kết quả -> ĐÚNG. Giá trị đúng viết "
        "ở dạng tương đương (làm tròn hợp lý, phân số/thập phân/phần trăm, đổi "
        "đơn vị) -> ĐÚNG. Nếu câu hỏi yêu cầu NHIỀU đại lượng thì phải đúng "
        "tất cả. Nếu câu hỏi hỏi đoạn văn có nêu một thông tin nào đó không, "
        "thì ĐÚNG là nói rõ đoạn văn KHÔNG nêu và cho biết đoạn văn thực sự "
        "cung cấp gì; SAI nếu bịa ra một con số.",
}


def build_prompt(task=None, language="vi"):
    """Base rubric, plus this task's criterion line when one is defined."""
    base = RUBRIC[language]
    line = TASK_CRITERIA_VI.get(task) if language == "vi" else None
    if not line:
        return base
    tail = ("\n\nCung cấp đánh giá" if language == "vi"
            else "\n\nProvide your judgment")
    return base.replace(tail, "\n" + line + tail, 1)
