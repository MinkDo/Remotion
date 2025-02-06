package com.example.hope.mood_tracker.ui

import android.annotation.SuppressLint
import android.os.Build
import androidx.annotation.RequiresApi
import androidx.compose.foundation.Image
import androidx.compose.foundation.gestures.detectVerticalDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.input.nestedscroll.NestedScrollConnection
import androidx.compose.ui.input.nestedscroll.NestedScrollSource
import androidx.compose.ui.input.nestedscroll.nestedScroll
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import com.example.hope.mood_tracker.data.Emotions
import com.example.hope.mood_tracker.ui.components.CalendarView
import com.example.hope.mood_tracker.ui.components.JournalEntryInput
import com.example.hope.mood_tracker.ui.components.MonthPickerDialog
import com.example.hope.mood_tracker.ui.components.MonthSelector
import java.time.LocalDate
import java.time.YearMonth
import java.time.format.TextStyle
import java.util.Locale

@SuppressLint("ReturnFromAwaitPointerEventScope")
@RequiresApi(Build.VERSION_CODES.O)
@Composable
fun NoteScreen(
    noteViewModel: NoteViewModel
) {
    val state: NoteState by noteViewModel.state.collectAsState()
    val onEvent = noteViewModel::onEvent
    val getEmotion = noteViewModel::getEmotion
    val getNoteByDate = noteViewModel::getNoteByDate

    var currentMonth by remember { mutableStateOf(YearMonth.now()) }
    val today = LocalDate.now()
    var isViewingEntry by remember { mutableStateOf(false) }
    var showMonthPicker by remember { mutableStateOf(false) }
    var isEditingNote by remember { mutableStateOf(false) }

    var isMonthChanged by remember { mutableStateOf(false) }

    Scaffold(
        floatingActionButton = {
            FloatingActionButton(
                onClick = {
                    onEvent(NoteEvent.SetDate(today))
                    onEvent(NoteEvent.ShowDialog)
                    isViewingEntry = getEmotion(today) != 0
                }
            ) {
                Icon(Icons.Filled.Add, contentDescription = "Add")
            }
        }
    ) { padding ->
        Column(modifier = Modifier
            .padding()
            .fillMaxHeight()
            .pointerInput(Unit) {
                detectVerticalDragGestures(
                    onVerticalDrag = { change, dragAmount ->
                        if (!isMonthChanged) { // Kiểm tra nếu chưa thay đổi tháng
                            if (dragAmount > 0) {
                                // Vuốt lên - Lùi về tháng trước
                                if (currentMonth > currentMonth.minusMonths(1)) {
                                    currentMonth = currentMonth.minusMonths(1)
                                    isMonthChanged = true // Đánh dấu đã thay đổi tháng
                                }
                            } else if (dragAmount < 0) {
                                // Vuốt xuống - Tiến tới tháng tiếp theo
                                if (currentMonth < YearMonth.now()) {
                                    currentMonth = currentMonth.plusMonths(1)
                                    isMonthChanged = true // Đánh dấu đã thay đổi tháng
                                }
                            }
                        }
                    },
                    onDragEnd = {
                        isMonthChanged = false // Reset lại sau khi kết thúc thao tác vuốt
                    }
                )
            }
        ) {
            MonthSelector(
                currentMonth = currentMonth,
                onPreviousMonth = { if (currentMonth > currentMonth.minusMonths(1)) currentMonth = currentMonth.minusMonths(1) },
                onNextMonth = { if (currentMonth < YearMonth.now()) currentMonth = currentMonth.plusMonths(1) },
                onMonthSelect = { showMonthPicker = true }
            )

            Spacer(modifier = Modifier.height(8.dp))

            Column(
                modifier = Modifier
                    .padding(padding)
                    .fillMaxHeight()
                    .nestedScroll(object : NestedScrollConnection {
                        override fun onPreScroll(available: Offset, source: NestedScrollSource): Offset {
                            if (!isMonthChanged) {
                                if (available.y > 70) { // Vuốt xuống
                                    if (currentMonth > currentMonth.minusMonths(1)) {
                                        currentMonth = currentMonth.minusMonths(1)
                                        isMonthChanged = true // Đánh dấu đã thay đổi tháng
                                    }
                                } else if (available.y < -70) { // Vuốt lên
                                    if (currentMonth < YearMonth.now()) {
                                        currentMonth = currentMonth.plusMonths(1)
                                        isMonthChanged = true // Đánh dấu đã thay đổi tháng
                                    }
                                }
                            }
                            return Offset.Zero
                        }

                        override fun onPostScroll(consumed: Offset, available: Offset, source: NestedScrollSource): Offset {
                            isMonthChanged = false // Reset lại sau khi kết thúc thao tác vuốt
                            return Offset.Zero
                        }
                    })
            ) {
                CalendarView(
                    getEmotion = getEmotion,
                    currentMonth = currentMonth,
                    today = today,
                    onDateSelected = { date ->
                        if (date <= today) {
                            onEvent(NoteEvent.SetDate(date))
                            onEvent(NoteEvent.ShowDialog)
                            isViewingEntry = getEmotion(date) != 0
                        }
                    },
                )
            }

            if (showMonthPicker) {
                MonthPickerDialog(
                    initialMonth = currentMonth,
                    onMonthSelected = { selectedMonth ->
                        currentMonth = selectedMonth
                        showMonthPicker = false
                    },
                    onDismiss = { showMonthPicker = false }
                )
            }

            // Hộp thoại cho việc thêm/xem nhật ký
            if (state.isAddingNote) {
                val note = getNoteByDate(state.date)
                AlertDialog(
                    onDismissRequest = {
                        onEvent(NoteEvent.HideDialog)
                        isEditingNote = false
                    },
                    title = {
                        Text(text = if (isViewingEntry && !isEditingNote) "Xem nhật ký"
                                    else if(!isEditingNote) "Thêm nhật ký"
                                    else "Chỉnh sửa nhật ký"
                        )
                    },
                    text = {
                        if (isViewingEntry && !isEditingNote) {
                            // Chế độ xem nhật ký
                            Column(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(4.dp), // Thêm padding xung quanh Column
                                horizontalAlignment = Alignment.CenterHorizontally, // Căn giữa nội dung theo chiều ngang
                                verticalArrangement = Arrangement.Center // Căn giữa nội dung theo chiều dọc
                            ) {

                                note?.date?.let {
                                    Text(
                                        text = "${it.dayOfMonth} ${it.month.getDisplayName(TextStyle.FULL, Locale.getDefault())}",
                                        style = MaterialTheme.typography.bodyLarge, // Chọn kiểu chữ phù hợp
                                        modifier = Modifier
                                            .padding(end = 16.dp) // Cách văn bản một chút với hình ảnh
                                            .align(Alignment.Start) // Căn giữa văn bản
                                    )
                                }
                                note?.emotion?.let {
                                    val imageResource = Emotions.images[it - 1]
                                    Image(
                                        painter = painterResource(id = imageResource),
                                        contentDescription = "Emotion $it",
                                        modifier = Modifier
                                            .size(100.dp) // Tăng kích thước hình ảnh để dễ nhìn hơn
                                            .padding(4.dp) // Thêm padding cho hình ảnh
                                    )
                                }

                                note?.content?.let {
                                    Text(
                                        text = it,
                                        style = MaterialTheme.typography.bodyLarge, // Chọn kiểu chữ phù hợp
                                        modifier = Modifier
                                            .padding(top = 16.dp) // Cách văn bản một chút với hình ảnh
                                            .align(Alignment.CenterHorizontally) // Căn giữa văn bản
                                    )
                                }
                            }
                        }
                        else {
                            // Chế độ thêm hoặc chỉnh sửa nhật ký
                            JournalEntryInput(
                                date = state.date,
                                content = state.content,
                                emotion = state.emotion,
                                onEntrySaved = {
                                    if (!isEditingNote) {
                                        onEvent(NoteEvent.SaveNote)
                                    } else {
                                        // ***Lưu ý chỗ này: Cập nhật note trước khi gửi vào sự kiện UpdateNote
                                        note?.let {
                                            val updatedNote = it.copy(
                                                content = state.content,
                                                emotion = state.emotion
                                            )
                                            onEvent(NoteEvent.UpdateNote(updatedNote))
                                        }
                                    }
                                    onEvent(NoteEvent.HideDialog)
                                    isEditingNote = false
                                },
                                onContentChange = { newContent ->
                                    onEvent(NoteEvent.SetContent(newContent))
                                },
                                onEmotionChange = { newEmotion ->
                                    onEvent(NoteEvent.SetEmotion(newEmotion))
                                }
                            )
                        }
                    },
                    confirmButton = {
                    },
                    dismissButton = {
                        if (isViewingEntry && !isEditingNote) {
                            TextButton(onClick = {
                                // Thay đổi sang chế độ chỉnh sửa
                                isEditingNote = true
                                if (note != null) {
                                    onEvent(NoteEvent.SetContent(note.content))
                                    onEvent(NoteEvent.SetEmotion(note.emotion))
                                } // Đặt nội dung hiện tại cho chỉnh sửa
                            }) {
                                Text("Sửa")
                            }

                            TextButton(onClick = {
                                // Xóa ghi chú
                                note?.let { NoteEvent.DeleteNote(it) }?.let { onEvent(it) }
                                onEvent(NoteEvent.HideDialog)
                            }) {
                                Text("Xóa")
                            }
                        }
                    }
                )
            }
        }
    }
}