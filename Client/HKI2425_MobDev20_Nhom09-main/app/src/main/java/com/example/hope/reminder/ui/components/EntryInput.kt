package com.example.hope.reminder.ui.components

import android.os.Build
import androidx.annotation.RequiresApi
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.DateRange
import androidx.compose.material.icons.filled.Notifications
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.rounded.Notifications
import androidx.compose.material3.Button
import androidx.compose.material3.Divider
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.hope.reminder.data.RepeatOption
import com.example.hope.reminder.ui.TaskEvent
import com.example.hope.reminder.ui.TaskViewModel
import com.vanpra.composematerialdialogs.MaterialDialog
import com.vanpra.composematerialdialogs.datetime.date.datepicker
import com.vanpra.composematerialdialogs.datetime.time.timepicker
import com.vanpra.composematerialdialogs.rememberMaterialDialogState
import java.time.LocalDate
import java.time.LocalTime
import java.time.format.DateTimeFormatter

@RequiresApi(Build.VERSION_CODES.O)
@Composable
fun EntryInput(
    viewModel: TaskViewModel
) {
    val state by viewModel.state.collectAsState()
    Column {
        // Tiêu đề Task
        TextField(
            value = state.title,
            onValueChange = { viewModel.onEvent(TaskEvent.SetTitle(it)) },
            label = { Text("Tiêu đề") }
        )
        Spacer(modifier = Modifier.height(8.dp))

        // Nội dung Task
        TextField(
            value = state.content ?: "",
            onValueChange = { viewModel.onEvent(TaskEvent.SetContent(it)) },
            label = { Text("Nội dung") }
        )
        Spacer(modifier = Modifier.height(8.dp))

        // Selector cho RepeatOption
        var expanded by remember { mutableStateOf(false) }
        Box {
            TextField(
                value = state.repeatOption.toString(),
                onValueChange = {},
                label = { Text("Lặp lại") },
                readOnly = true,
                modifier = Modifier.fillMaxWidth(),
                trailingIcon = {
                    IconButton(onClick = { expanded = true }) {
                        Icon(Icons.Default.ArrowDropDown, contentDescription = "Select Repeat Option")
                    }
                }
            )
            DropdownMenu(
                expanded = expanded,
                onDismissRequest = { expanded = false }
            ) {
                DropdownMenuItem(
                    text = { Text("Không lặp lại") },
                    onClick = {
                        viewModel.onEvent(TaskEvent.SetRepeatOption(RepeatOption.NONE))
                        expanded = false
                    }
                )
                DropdownMenuItem(
                    text = { Text("Lặp lại hằng ngày") },
                    onClick = {
                        viewModel.onEvent(TaskEvent.SetRepeatOption(RepeatOption.DAILY))
                        expanded = false
                    }
                )
                DropdownMenuItem(
                    text = { Text("Lặp lại theo tuần") },
                    onClick = {
                        viewModel.onEvent(TaskEvent.SetRepeatOption(RepeatOption.DAYINWEEK))
                        expanded = false
                    }
                )
                DropdownMenuItem(
                    text = { Text("Lặp lại theo tháng") },
                    onClick = {
                        viewModel.onEvent(TaskEvent.SetRepeatOption(RepeatOption.MONTHLY))
                        expanded = false
                    }
                )
                DropdownMenuItem(
                    text = { Text("Lặp lại theo năm") },
                    onClick = {
                        viewModel.onEvent(TaskEvent.SetRepeatOption(RepeatOption.YEARLY))
                        expanded = false
                    }
                )
            }
        }
        Spacer(modifier = Modifier.height(8.dp))

        // Chọn giờ thực hiện
        val timeDialogState = rememberMaterialDialogState()
        Column {
            Text(text = "Chọn giờ")
            Row(
                    modifier = Modifier
                        .padding(top = 2.dp)
                        .clickable { timeDialogState.show() }
            ) {
            Text(
                text = state.time?.format(DateTimeFormatter.ofPattern("HH:mm")) ?: "--:--",
                style = TextStyle(
                    fontFamily = FontFamily.SansSerif,
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Medium
                )
            )
            Icon(
                imageVector = Icons.Default.Notifications,
                contentDescription = null,
                modifier = Modifier
                    .padding(start = 4.dp)
            )
        }
        }

        Spacer(modifier = Modifier.height(8.dp))

        MaterialDialog(
            dialogState = timeDialogState,
            buttons = {
                positiveButton(text = "Ok") {
                }
                negativeButton(text = "Cancel")
            }
        ) {
            timepicker(
                initialTime = LocalTime.now(),
                title = "Pick a time"
            ) {
                viewModel.onEvent(TaskEvent.SetTime(it))
            }
        }

        // datepicker startdate
        val startDateDialogState = rememberMaterialDialogState()
        MaterialDialog(
            dialogState = startDateDialogState,
            buttons = {
                positiveButton(text = "Ok") {
                }
                negativeButton(text = "Cancel")
            }
        ) {
            datepicker(
                initialDate = state.startDate ?: LocalDate.now(),
                title = "Pick a date",
                allowedDateValidator = {
                    it >= LocalDate.now()
                }
            ) {
                viewModel.onEvent(TaskEvent.SetStartDate(it))
            }
        }
        Row {
            Column {
                Text(text = "Chọn ngày bắt đầu")
                Row(
                    modifier = Modifier
                        .padding(top = 2.dp)
                        .clickable { startDateDialogState.show() },
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = state.startDate?.format(DateTimeFormatter.ofPattern("dd/MM/yyyy")) ?: "--/--/----",
                        style = TextStyle(
                            fontFamily = FontFamily.SansSerif,
                            fontSize = 20.sp,
                            fontWeight = FontWeight.Medium
                        )
                    )
                    Icon(
                        imageVector = Icons.Default.DateRange,
                        contentDescription = null,
                        modifier = Modifier
                            .padding(start = 4.dp)
                    )
                }

            }
            Spacer(modifier = Modifier.weight(1f))
            if (state.repeatOption !== RepeatOption.NONE) {

                val endDateDialogState = rememberMaterialDialogState()
                MaterialDialog(
                    dialogState = endDateDialogState,
                    buttons = {
                        positiveButton(text = "Ok") {
                        }
                        negativeButton(text = "Cancel")
                    }
                ) {
                    datepicker(
                        initialDate = state.startDate?.plusDays(1) ?: LocalDate.now(),
                        title = "Pick end date",
                        allowedDateValidator = {
                            it > state.startDate
                        }
                    ) {
                        viewModel.onEvent(TaskEvent.SetEndDate(it))
                    }
                }
                Column {
                    Text(text = "Chọn ngày kết thúc")
                    Row(
                        modifier = Modifier
                            .padding(top = 2.dp)
                            .clickable { endDateDialogState.show() },
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = state.endDate?.format(DateTimeFormatter.ofPattern("dd/MM/yyyy")) ?: "--/--/----",
                            style = TextStyle(
                                fontFamily = FontFamily.SansSerif,
                                fontSize = 20.sp,
                                fontWeight = FontWeight.Medium
                            )
                        )
                        Icon(
                            imageVector = Icons.Default.DateRange,
                            contentDescription = null,
                            modifier = Modifier
                                .padding(start = 4.dp)
                        )
                    }

                }
            }
        }

        // Chọn ngày bắt đầu và kết thúc (chỉ hiển thị khi lặp lại)
        if (state.repeatOption == RepeatOption.DAYINWEEK) {
            Divider(
                thickness = 1.dp,
                modifier = Modifier
                    .padding(top = 8.dp)
            )
            Column(
                modifier = Modifier.padding(top = 4.dp)
            ) {
                Text("Chọn các ngày trong tuần")

                // Tạo các Text trong Row để chọn các ngày trong tuần
                val daysOfWeek = listOf("CN", "T2", "T3", "T4", "T5", "T6", "T7")
                val daysIndex = daysOfWeek.indices.toList()

                // Row cho các ngày trong tuần
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                ) {
                    daysIndex.forEach { index ->
                        val isSelected = state.selectedDaysOfWeek.contains(index)

                        // Button với border khi chọn
                        Box(
                            modifier = Modifier
                                .padding(2.dp)
                                .border(
                                    width = 1.dp,
                                    color = if (isSelected) androidx.compose.ui.graphics.Color.Black else androidx.compose.ui.graphics.Color.Gray,
                                    shape = androidx.compose.ui.graphics.RectangleShape
                                )
                                .clickable {
                                    // Thêm hoặc bỏ chọn ngày
                                    val newSelectedDays = if (isSelected) {
                                        state.selectedDaysOfWeek - index
                                    } else {
                                        state.selectedDaysOfWeek + index
                                    }
                                    viewModel.onEvent(TaskEvent.SetSelectedDaysOfWeek(newSelectedDays))
                                }
                                .padding(8.dp)
                                .weight(1f)
                        ) {
                            Text(
                                text = daysOfWeek[index],
                                style = androidx.compose.ui.text.TextStyle(
                                    fontWeight = if (isSelected) androidx.compose.ui.text.font.FontWeight.Bold else androidx.compose.ui.text.font.FontWeight.Normal
                                ),
                                color = androidx.compose.ui.graphics.Color.Black
                            )
                        }
                    }
                }

                // Hiển thị các ngày đã chọn
                if (state.selectedDaysOfWeek.isNotEmpty()) {
                    Text("Ngày đã chọn: " + state.selectedDaysOfWeek.joinToString(", ") { daysOfWeek[it] })
                } else {
                    Text("Chưa chọn ngày nào.")
                }
            }
        }
    }
}