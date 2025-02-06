package com.example.hope.mood_tracker.ui.components

import android.os.Build
import androidx.annotation.RequiresApi
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.hope.mood_tracker.data.Emotions
import java.time.LocalDate

@RequiresApi(Build.VERSION_CODES.O)
@Composable
fun DayView(
    date: LocalDate,
    emotion: Int,
    isFutureDate: Boolean,
    onClick: () -> Unit
) {
    Box(
        modifier = Modifier
            .size(54.dp)
            .padding(2.dp)
            .clickable(enabled = !isFutureDate) { onClick() },
        contentAlignment = Alignment.Center
    ) {
        if(emotion > 0) {
            val imageResource = Emotions.images[emotion-1]
            Image(
                painter = painterResource(id = imageResource),
                contentDescription = "Emotion $emotion",
                modifier = Modifier
                    .size(70.dp) // Kích thước hình ảnh
                    .padding(2.dp)
            )
        }
        else {
            Text(
                text = date.dayOfMonth.toString(),
                fontSize = 16.sp,
                fontWeight = if (!isFutureDate) FontWeight.Bold else FontWeight.Thin
            )
        }
    }
}
