package com.example.hope.dass21.data.model

// Model để gửi dữ liệu form
data class FormData(
    val q1: Int,
    val q2: Int,
    val q3: Int,
    val q4: Int,
    val q5: Int,
    val q6: Int,
    val q7: Int,
    val q8: Int,
    val q9: Int,
    val q10: Int,
    val q11: Int,
    val q12: Int,
    val q13: Int,
    val q14: Int,
    val q15: Int,
    val q16: Int,
    val q17: Int,
    val q18: Int,
    val q19: Int,
    val q20: Int,
    val q21: Int
)

// Model để nhận kết quả từ API
data class ResultResponse(
    val anxiety_score: Double,
    val depression_score: Double,
    val stress_score: Double,
    val calculation_success: Boolean
)


data class PredictForm(
    val text: String
)
data class PredictResponse(
    val predicted_emotion: String
)