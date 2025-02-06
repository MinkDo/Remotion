package com.example.hope.mood_tracker.ui.components

import android.os.Build
import androidx.annotation.RequiresApi
import androidx.compose.foundation.gestures.detectVerticalDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import java.time.YearMonth
import java.time.format.DateTimeFormatter

@RequiresApi(Build.VERSION_CODES.O)
@Composable
fun MonthPickerDialog(
    initialMonth: YearMonth,
    onMonthSelected: (YearMonth) -> Unit,
    onDismiss: () -> Unit
) {
    var selectedMonth by remember { mutableStateOf(initialMonth) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Chọn tháng") },
        text = {
            Box(modifier = Modifier.height(150.dp)) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center,
                    modifier = Modifier
                        .fillMaxWidth()
                        .pointerInput(Unit) {
                            detectVerticalDragGestures { _, dragAmount ->
                                if (dragAmount < -15 && selectedMonth < YearMonth.now()) { // Vuốt lên để chuyển tháng trước
                                    selectedMonth = selectedMonth.plusMonths(1)
                                } else if (dragAmount > 15) { // Vuốt xuống để chuyển tháng sau
                                    selectedMonth = selectedMonth.minusMonths(1)
                                }
                            }
                        }
                ) {
                    // Tháng liền trước, hiện mờ
                    Text(
                        text = selectedMonth.minusMonths(1).format(DateTimeFormatter.ofPattern("MMMM yyyy")),
                        style = TextStyle(fontSize = 24.sp),
                        color = Color.Gray.copy(alpha = 0.5f)
                    )
                    // Tháng hiện tại
                    Text(
                        text = selectedMonth.format(DateTimeFormatter.ofPattern("MMMM yyyy")),
                        style = TextStyle(fontSize = 32.sp),
                        fontWeight = FontWeight.Bold
                    )
                    // Tháng liền sau, hiện mờ
                    Text(
                        text = selectedMonth.plusMonths(1).format(DateTimeFormatter.ofPattern("MMMM yyyy")),
                        style = TextStyle(fontSize = 24.sp),
                        color = Color.Gray.copy(alpha = 0.5f)
                    )
                }
            }
        },
        confirmButton = {
            TextButton(onClick = { onMonthSelected(selectedMonth) }) {
                Text("Chọn")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Hủy")
            }
        }
    )
}