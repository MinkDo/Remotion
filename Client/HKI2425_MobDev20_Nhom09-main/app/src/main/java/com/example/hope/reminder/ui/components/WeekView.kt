package com.example.hope.reminder.ui.components

import android.os.Build
import androidx.annotation.RequiresApi
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import java.time.LocalDate

@RequiresApi(Build.VERSION_CODES.O)
@Composable
fun WeekView(
    startOfWeek: LocalDate,
    selectedDate: LocalDate?,
    onDateSelected: (LocalDate) -> Unit
) {
    // Điều chỉnh startOfWeek về đúng ngày Chủ Nhật nếu không phải.
    val adjustedStartOfWeek = startOfWeek.minusDays((startOfWeek.dayOfWeek.value % 7).toLong())

    // Lấy danh sách các ngày trong tuần
    val daysInWeek = (0..6).map { adjustedStartOfWeek.plusDays(it.toLong()) }

    // Tiêu đề các ngày trong tuần (CN, T2, ...)
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(start = 0.dp, top = 0.dp, end = 0.dp, bottom = 16.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        listOf("CN", "T2", "T3", "T4", "T5", "T6", "T7").forEach { dayOfWeek ->
            Text(
                text = dayOfWeek,
                modifier = Modifier.weight(1f),
                textAlign = androidx.compose.ui.text.style.TextAlign.Center
            )
        }
    }

    // Hiển thị ngày trong tuần
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        daysInWeek.forEach { date ->
            DayView(
                date = date,
                isSelected = selectedDate == date,
                onClick = { onDateSelected(date) }
            )
        }
    }
}