package com.example.hope.mood_tracker.ui.components

import android.os.Build
import androidx.annotation.RequiresApi
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.hope.R
import com.example.hope.mood_tracker.data.Emotions
import java.time.LocalDate
import java.time.format.TextStyle
import java.util.Locale

@RequiresApi(Build.VERSION_CODES.O)
@Composable
fun JournalEntryInput(
    date: LocalDate,
    content: String,
    emotion: Int,
    onEntrySaved: () -> Unit,
    onContentChange: (String) -> Unit,
    onEmotionChange: (Int) -> Unit
) {
//    val emotions = listOf(
//        R.drawable.happy,  // Hình ảnh vui vẻ
//        R.drawable.relax, // Hình ảnh thư giãn
//        R.drawable.bored,   // Hình ảnh chán nản
//        R.drawable.sad,     // Hình ảnh buồn
//        R.drawable.angry    // Hình ảnh tức giận
//    )

    Column {
        Text(
            text = "Nhật ký ngày ${date.dayOfMonth} ${date.month.getDisplayName(TextStyle.FULL, Locale.getDefault())}",
            style = MaterialTheme.typography.bodyLarge,
            fontWeight = FontWeight.Bold
        )

        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceEvenly // Căn giữa các hình ảnh và phân bổ không gian đều
        ) {
            for (emotionIndex in 1..5) {
                val imageResource = Emotions.images[emotionIndex - 1]
                val isSelected = emotion == emotionIndex

                Image(
                    painter = painterResource(id = imageResource),
                    contentDescription = "Emotion $emotionIndex",
                    modifier = Modifier
                        .size(70.dp) // Kích thước hình ảnh
                        .alpha(if (isSelected) 1f else 0.3f) // Đậm hơn khi được chọn
                        .clickable { onEmotionChange(emotionIndex) } // Thực hiện hành động khi click
                        .padding(2.dp)
                        .weight(1f) // Đảm bảo mỗi hình chiếm cùng một tỷ lệ không gian
                )
            }
        }


        OutlinedTextField(
            value = content,
            onValueChange = { onContentChange(it) }, // Gọi hàm để thay đổi nội dung
            modifier = Modifier.fillMaxWidth(),
            label = { Text("Nhập nội dung nhật ký") }
        )
        Spacer(modifier = Modifier.height(8.dp))
        Button(
            onClick = {
                onEntrySaved()
                onContentChange("") // Reset lại `content` khi lưu
            },
            modifier = Modifier.align(Alignment.End).padding(top = 16.dp)
        ) {
            Text("Lưu")
        }
    }
}