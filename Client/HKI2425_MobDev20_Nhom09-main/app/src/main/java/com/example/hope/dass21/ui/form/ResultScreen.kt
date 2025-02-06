package com.example.hope.dass21.ui.form

import android.widget.Space
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.hope.dass21.data.model.ResultResponse


@Composable
fun ResultScreen(
    resultResponse: ResultResponse?,
    onNextStage: () -> Unit,
    onReturnHome: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {


        Spacer(modifier = Modifier.height(8.dp))

        // Result Section
        LazyColumn(
            modifier = Modifier.weight(1f)
        ) {
            item {
                if (resultResponse != null) {
                    ResultCard(
                        title = "Chỉ số lo âu",
                        score = resultResponse.anxiety_score.toInt(),
                        message = getAnxietyMessage(resultResponse.anxiety_score.toInt())
                    )
                }
            }
            item {
                if (resultResponse != null) {
                    ResultCard(
                        title = "Chỉ số trầm cảm",
                        score = resultResponse.depression_score.toInt(),
                        message = getDepressionMessage(resultResponse.depression_score.toInt())
                    )
                }
            }
            item {
                if (resultResponse != null) {
                    ResultCard(
                        title = "Chỉ số stress",
                        score = resultResponse.stress_score.toInt(),
                        message = getStressMessage(resultResponse.stress_score.toInt())
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Next Stage Button
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(modifier = Modifier.fillMaxWidth()) {
                Button(
                    onClick = onNextStage,
                    modifier = Modifier
                        .weight(1f) // Chia đều không gian
                        .fillMaxWidth() // Đảm bảo button chiếm toàn bộ phần không gian được cấp
                ) {
                    Text(text = "Làm lại")
                }

                Spacer(modifier = Modifier.width(4.dp))

                Button(
                    onClick = onReturnHome,
                    modifier = Modifier
                        .weight(1f) // Chia đều không gian
                        .fillMaxWidth() // Đảm bảo button chiếm toàn bộ phần không gian được cấp
                ) {
                    Text(text = "Quay về")
                }
            }

        }
    }
}

@Composable
fun ResultCard(title: String, score: Int, message: String) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),

        ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = title,
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary
            )
            Text(
                text = "Điểm: $score",
                modifier = Modifier.padding(top = 6.dp)
                )
            Text(
                text = message,
                modifier = Modifier.padding(top = 2.dp)
            )
        }
    }
}

// Helper Functions to Get Messages
fun getAnxietyMessage(score: Int): String {
    return when {
        score < 8 -> "Mức độ lo âu của bạn là bình thường. Hãy duy trì lối sống hiện tại của bạn."
        score < 10 -> "Lo âu nhẹ. Hãy thử áp dụng các kỹ thuật thư giãn."
        score < 15 -> "Lo âu trung bình. Cân nhắc tập thể dục thường xuyên và thực hành chánh niệm."
        score < 20 -> "Lo âu nghiêm trọng. Hãy tìm sự hỗ trợ chuyên môn nếu cần thiết."
        else -> "Lo âu cực kỳ nghiêm trọng. Hãy tham khảo ý kiến chuyên gia tâm lý."
    }
}

fun getDepressionMessage(score: Int): String {
    return when {
        score < 10 -> "Mức độ trầm cảm của bạn là bình thường. Hãy duy trì sự kết nối xã hội."
        score < 14 -> "Trầm cảm nhẹ. Hãy thử đặt ra những mục tiêu nhỏ có thể đạt được."
        score < 21 -> "Trầm cảm trung bình. Hãy tham gia các hoạt động thể chất và sở thích cá nhân."
        score < 28 -> "Trầm cảm nghiêm trọng. Hãy kết nối với một chuyên gia tư vấn để được hỗ trợ."
        else -> "Trầm cảm cực kỳ nghiêm trọng. Hãy tìm kiếm sự trợ giúp ngay lập tức."
    }
}

fun getStressMessage(score: Int): String {
    return when {
        score < 10 -> "Mức độ căng thẳng của bạn đang được kiểm soát. Hãy duy trì các thói quen lành mạnh."
        score < 14 -> "Căng thẳng nhẹ. Hãy thực hành thiền và nghỉ ngơi thường xuyên."
        score < 21 -> "Căng thẳng trung bình. Dành thời gian hòa mình vào thiên nhiên và tập trung vào hơi thở."
        score < 28 -> "Căng thẳng nghiêm trọng. Hãy tìm lời khuyên từ chuyên gia để quản lý căng thẳng."
        else -> "Căng thẳng cực kỳ nghiêm trọng. Cần có sự can thiệp ngay lập tức."
    }
}