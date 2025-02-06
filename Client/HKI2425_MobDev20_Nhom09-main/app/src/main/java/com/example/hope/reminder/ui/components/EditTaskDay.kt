//package com.example.hope.reminder.ui.components
//
//import android.os.Build
//import androidx.annotation.RequiresApi
//import androidx.compose.foundation.clickable
//import androidx.compose.foundation.layout.Column
//import androidx.compose.foundation.layout.Row
//import androidx.compose.foundation.layout.Spacer
//import androidx.compose.foundation.layout.height
//import androidx.compose.foundation.layout.padding
//import androidx.compose.material.icons.Icons
//import androidx.compose.material.icons.filled.Notifications
//import androidx.compose.material3.Icon
//import androidx.compose.material3.Text
//import androidx.compose.material3.TextField
//import androidx.compose.runtime.Composable
//import androidx.compose.runtime.collectAsState
//import androidx.compose.runtime.getValue
//import androidx.compose.runtime.mutableStateOf
//import androidx.compose.runtime.remember
//import androidx.compose.runtime.setValue
//import androidx.compose.ui.Modifier
//import androidx.compose.ui.text.TextStyle
//import androidx.compose.ui.text.font.FontFamily
//import androidx.compose.ui.text.font.FontWeight
//import androidx.compose.ui.unit.dp
//import androidx.compose.ui.unit.sp
//import com.example.hope.reminder.ui.TaskEvent
//import com.example.hope.reminder.ui.TaskViewModel
//import com.vanpra.composematerialdialogs.MaterialDialog
//import com.vanpra.composematerialdialogs.datetime.time.timepicker
//import com.vanpra.composematerialdialogs.rememberMaterialDialogState
//import java.time.LocalTime
//import java.time.format.DateTimeFormatter
//
//@RequiresApi(Build.VERSION_CODES.O)
//@Composable
//fun EditTaskDay(
//    time: LocalTime?,
//    content: String?,
//    viewModel: TaskViewModel
//) {
//    val timeCopy by remember { mutableStateOf(time) }
//    var contentCopy by remember { mutableStateOf(content) }
//    val state by viewModel.state.collectAsState()
//    val onEvent = viewModel::onEvent
//
//    val timeDialogState = rememberMaterialDialogState()
//    Column {
//        Text(text = "Thời gian")
//        Row(
//            modifier = Modifier
//                .padding(top = 2.dp)
//                .clickable { timeDialogState.show() }
//        ) {
//            Text(
//                text = state.time?.format(DateTimeFormatter.ofPattern("HH:mm")) ?:
//                     (timeCopy?.format(DateTimeFormatter.ofPattern("HH:mm"))?: "--:--"),
//                style = TextStyle(
//                    fontFamily = FontFamily.SansSerif,
//                    fontSize = 24.sp,
//                    fontWeight = FontWeight.Medium
//                )
//            )
//            Icon(
//                imageVector = Icons.Default.Notifications,
//                contentDescription = null,
//                modifier = Modifier
//                    .padding(start = 4.dp)
//            )
//        }
//    }
//    MaterialDialog(
//        dialogState = timeDialogState,
//        buttons = {
//            positiveButton(text = "Ok") {
//            }
//            negativeButton(text = "Cancel")
//        }
//    ) {
//        timepicker(
//            initialTime = LocalTime.now(),
//            title = "Pick a time"
//        ) {
//            onEvent(TaskEvent.SetTime(it))
//        }
//    }
//
//    Text(text = "Nội dung")
//    TextField(
//        value = state.content?.takeIf { it.isNotEmpty() } ?: (contentCopy ?: ""),
//        onValueChange = {
//            onEvent(TaskEvent.SetContent(it))
//            contentCopy = ""
//        },
//        label = { Text("Nội dung") }
//    )
//
//    Spacer(modifier = Modifier.height(8.dp))
//}

package com.example.hope.reminder.ui.components

import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Notifications
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.hope.reminder.ui.TaskEvent
import com.example.hope.reminder.ui.TaskViewModel
import com.vanpra.composematerialdialogs.MaterialDialog
import com.vanpra.composematerialdialogs.datetime.time.timepicker
import com.vanpra.composematerialdialogs.rememberMaterialDialogState
import java.time.LocalTime
import java.time.format.DateTimeFormatter

@RequiresApi(Build.VERSION_CODES.O)
@Composable
fun EditTaskDay(
    time: LocalTime?,
    content: String?,
    viewModel: TaskViewModel
) {
    val state = viewModel.state.collectAsState()
    val onEvent = viewModel::onEvent

    // phải sử dụng LaunchedEffect nha
    LaunchedEffect(time) {
        if (state.value.time != time) {
            onEvent(TaskEvent.SetTime(time))
        }
    }
    LaunchedEffect(content) {
        if (state.value.content != content) {
            onEvent(TaskEvent.SetContent(content))
        }
    }
    Log.d("Show state", "${state.value.time}, ${state.value.content}")

    val timeDialogState = rememberMaterialDialogState()
    Column {
        Text(text = "Thời gian")
        Row(
            modifier = Modifier
                .padding(top = 2.dp)
                .clickable { timeDialogState.show() }
        ) {
            Text(
                text = state.value.time?.format(DateTimeFormatter.ofPattern("HH:mm")) ?: "--:--",
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
            onEvent(TaskEvent.SetTime(it))
        }
    }

    Text(text = "Nội dung")
    TextField(
        value = state.value.content ?: "",
        onValueChange = {
            onEvent(TaskEvent.SetContent(it))
        }
    )

    Spacer(modifier = Modifier.height(8.dp))
}