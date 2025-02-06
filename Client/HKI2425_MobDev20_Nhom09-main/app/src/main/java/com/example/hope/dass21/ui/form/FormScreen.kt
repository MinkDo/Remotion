package com.example.hope.dass21.ui.form

import android.util.Log
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.example.hope.dass21.data.model.FormData
import com.example.hope.dass21.ui.viewmodel.SurveyViewModel


@Composable
fun StressAnxietyDepressionForm(viewModel: SurveyViewModel, onSubmit: () -> Unit, onHomeBack: ()->Unit) {
    val questions = listOf(
        "Tôi nhận thức được miệng mình bị khô.",
        "Tôi không thể cảm nhận được bất kỳ cảm giác tích cực nào.",
        "Tôi cảm thấy khó để thư giãn.",
        "Tôi trải qua khó khăn trong việc thở (ví dụ: thở quá nhanh).",
        "Tôi cảm thấy khó để có động lực làm bất cứ việc gì.",
        "Tôi có xu hướng phản ứng thái quá với các tình huống.",
        "Tôi cảm thấy run rẩy (ví dụ: ở tay).",
        "Tôi cảm thấy rằng mình không có điều gì để mong đợi.",
        "Tôi cảm thấy mình đang sử dụng rất nhiều năng lượng thần kinh.",
        "Tôi lo lắng về các tình huống mà mình có thể hoảng sợ và trở nên ngớ ngẩn.",
        "Tôi cảm thấy chán nản và buồn bã.",
        "Tôi nhận thấy mình trở nên kích động.",
        "Tôi cảm thấy mình gần như hoảng loạn.",
        "Tôi không thể cảm thấy hào hứng về bất kỳ điều gì.",
        "Tôi cảm thấy khó thư giãn.",
        "Tôi nhận thức được nhịp tim của mình mà không có bất kỳ vận động nào.",
        "Tôi cảm thấy mình không có giá trị nhiều với tư cách là một con người.",
        "Tôi không thể chịu được bất kỳ điều gì làm cản trở công việc của mình.",
        "Tôi cảm thấy sợ hãi mà không có lý do chính đáng.",
        "Tôi cảm thấy rằng cuộc sống không có ý nghĩa.",
        "Tôi cảm thấy mình khá nhạy cảm."
    )

    val sliderValues = remember { mutableStateListOf<Float>().apply { addAll(List(questions.size) { 0f }) } }

    fun submitForm() {
        val formData = FormData(
            q1 = sliderValues[0].toInt(),
            q2 = sliderValues[1].toInt(),
            q3 = sliderValues[2].toInt(),
            q4 = sliderValues[3].toInt(),
            q5 = sliderValues[4].toInt(),
            q6 = sliderValues[5].toInt(),
            q7 = sliderValues[6].toInt(),
            q8 = sliderValues[7].toInt(),
            q9 = sliderValues[8].toInt(),
            q10 = sliderValues[9].toInt(),
            q11 = sliderValues[10].toInt(),
            q12 = sliderValues[11].toInt(),
            q13 = sliderValues[12].toInt(),
            q14 = sliderValues[13].toInt(),
            q15 = sliderValues[14].toInt(),
            q16 = sliderValues[15].toInt(),
            q17 = sliderValues[16].toInt(),
            q18 = sliderValues[17].toInt(),
            q19 = sliderValues[18].toInt(),
            q20 = sliderValues[19].toInt(),
            q21 = sliderValues[20].toInt()
        )
        // Log the formData
        Log.d("SubmitForm", "Form Data: $formData")
        viewModel.submitForm(formData)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        // Title
        Text(
            text = "Bộ câu hỏi Taylor Scale cho các mức độ Stress, Lo âu và Trầm cảm",
            color = MaterialTheme.colorScheme.primary
        )

        Spacer(modifier = Modifier.height(8.dp))

        // LazyColumn for Questions and Sliders
        LazyColumn(
            modifier = Modifier.weight(1f)
        ) {
            itemsIndexed(questions) { index, question ->
                QuestionSlider(
                    index = index,
                    question = question,
                    sliderValue = sliderValues[index],
                    onValueChange = { sliderValues[index] = it }
                )
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Submit Button
        Button(
            onClick = {
                onSubmit()
                submitForm()
            },
            modifier = Modifier.align(Alignment.CenterHorizontally)
        ) {
            Text(text = "Gửi", modifier = Modifier.fillMaxWidth(), textAlign = TextAlign.Center)
        }
    }
}

@Composable
fun QuestionSlider(index: Int, question: String, sliderValue: Float, onValueChange: (Float) -> Unit) {
    Column(modifier = Modifier.padding(vertical = 8.dp)) {
        Text(text = "${index + 1}. $question")
        Slider(
            value = sliderValue,
            onValueChange = onValueChange,
            valueRange = 0f..3f,
            steps = 2,
            modifier = Modifier.fillMaxWidth()
        )
    }
}