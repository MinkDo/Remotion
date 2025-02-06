package com.example.hope.reminder.ui

import android.os.Build
import androidx.annotation.RequiresApi
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.hope.reminder.data.RepeatOption
import com.example.hope.reminder.data.database.Task
import com.example.hope.reminder.data.database.TaskDay
import com.example.hope.reminder.ui.components.EntryInput
import com.example.hope.reminder.ui.components.TaskList
import com.example.hope.reminder.ui.components.WeekSelector
import com.example.hope.reminder.ui.components.WeekView
import java.time.LocalDate

@RequiresApi(Build.VERSION_CODES.O)
@Composable
fun TaskScreen(viewModel: TaskViewModel) {

    val state by viewModel.state.collectAsState()
    val onEvent = viewModel::onEvent

    var startOfWeek by remember {
        mutableStateOf(
            LocalDate.now().minusDays((LocalDate.now().dayOfWeek.value % 7).toLong())
        )
    }

    var showDialog by remember { mutableStateOf(false) }
    var missingFields by remember { mutableStateOf<List<String>>(emptyList()) }

    fun handleSave() {
        val missing = mutableListOf<String>()
        if (state.title.isBlank()) missing.add("Tiêu đề")
        if (state.startDate == null) missing.add("Ngày bắt đầu")
        if (state.repeatOption!==RepeatOption.NONE && state.endDate == null) missing.add("Ngày kết thúc")
        if(state.repeatOption==RepeatOption.DAYINWEEK && state.selectedDaysOfWeek.isEmpty()) missing.add("Ít nhất 1 ngày trong tuần")

        // Nếu có mục còn thiếu, hiển thị dialog
        if (missing.isNotEmpty()) {
            missingFields = missing
            showDialog = true
        } else {
            onEvent(TaskEvent.SaveTask)
        }
    }

    if (showDialog) {
        AlertDialog(
            onDismissRequest = { showDialog = false },
            title = { Text(text = "Thông báo", fontWeight = FontWeight.Bold) },
            text = {
                Text(
                    text = "Bạn cần nhập đầy đủ các trường:\n - ${missingFields.joinToString("\n - ")}",
                    fontWeight = FontWeight.Normal
                )
            },
            confirmButton = {
                TextButton(onClick = { showDialog = false }) {
                    Text(text = "OK")
                }
            }
        )
    }


    Scaffold(
        floatingActionButton = {
            FloatingActionButton(
                onClick = { onEvent(TaskEvent.ShowDialog) }
            ) {
                Icon(Icons.Filled.Add, contentDescription = "Add")
            }
        }
    ) { padding ->
        Column(
            modifier = Modifier.fillMaxHeight()
        ) {
            WeekSelector(
                startOfWeek = startOfWeek,
                onPreviousWeek = {
                    startOfWeek = startOfWeek.minusWeeks(1)
                },
                onNextWeek = {
                    startOfWeek = startOfWeek.plusWeeks(1)
                }
            )

            Column(
                modifier = Modifier
                    .padding(padding)
                    .fillMaxHeight()
            ) {
                WeekView(
                    startOfWeek = startOfWeek,
                    selectedDate = state.dateSelected,
                    onDateSelected = { date ->
                        onEvent(TaskEvent.SetDateSelected(date))
                    }
                )
                Spacer(modifier = Modifier.height(8.dp))
                var taskList by remember { mutableStateOf<List<Pair<Task, TaskDay>>>(emptyList()) }
                LaunchedEffect(state.tasks, state.dateSelected) {
                    taskList = viewModel.getTaskByDate(state.dateSelected)
                }

                TaskList(
                    tasks = taskList,
                    viewModel = viewModel
                )
            }
        }
        if (state.isAddingTask) {
            AlertDialog(
                onDismissRequest = { onEvent(TaskEvent.HideDialog) },
                title = { Text("Thêm Task") },
                text = {
                    EntryInput(viewModel)
                },
                confirmButton = {
                    Button(
                        onClick = { handleSave() }
                    ) {
                        Text("Lưu")
                    }
                },
                dismissButton = {
                    Button(onClick = { onEvent(TaskEvent.HideDialog) }) {
                        Text("Hủy")
                    }
                }
            )
        }
    }
}