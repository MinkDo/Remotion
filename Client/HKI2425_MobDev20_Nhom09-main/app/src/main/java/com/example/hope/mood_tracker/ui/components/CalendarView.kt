package com.example.hope.mood_tracker.ui.components

import android.os.Build
import androidx.annotation.RequiresApi
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import java.time.LocalDate
import java.time.YearMonth

@RequiresApi(Build.VERSION_CODES.O)
@Composable
fun CalendarView(
    getEmotion: (LocalDate) -> Int,
    currentMonth: YearMonth,
    today: LocalDate,
    onDateSelected: (LocalDate) -> Unit
) {
    val daysInMonth = (1..currentMonth.lengthOfMonth()).map { day ->
        currentMonth.atDay(day)
    }

    Row(modifier = Modifier.fillMaxWidth()
        .padding(start = 0.dp,top=0.dp, end = 0.dp, bottom=30.dp),
        horizontalArrangement = Arrangement.SpaceBetween) {
        listOf("CN", "T2", "T3", "T4", "T5", "T6", "T7").forEach { dayOfWeek ->
            Text(text = dayOfWeek, modifier = Modifier.weight(1f), textAlign = TextAlign.Center)
        }
    }

    LazyVerticalGrid(
        columns = GridCells.Fixed(7),
        contentPadding = PaddingValues(4.dp)
    ) {
        val firstDayOfWeek = currentMonth.atDay(1).dayOfWeek.value
        items(firstDayOfWeek - 1) {
            Box(modifier = Modifier.size(48.dp))
        }

        items(daysInMonth) { date ->
            val emotion = getEmotion(date)
            val isFutureDate = date > today
            DayView(
                date = date,
                emotion = emotion,
                isFutureDate = isFutureDate,
                onClick = { if (!isFutureDate) onDateSelected(date) }
            )
        }
    }
}