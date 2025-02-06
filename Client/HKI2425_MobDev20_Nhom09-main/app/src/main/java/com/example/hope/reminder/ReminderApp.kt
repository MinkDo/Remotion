package com.example.hope.reminder

import android.annotation.SuppressLint
import android.os.Build
import androidx.annotation.RequiresApi
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.hope.reminder.ui.TaskScreen
import com.example.hope.reminder.ui.TaskViewModel

@SuppressLint("UnusedMaterial3ScaffoldPaddingParameter")
@RequiresApi(Build.VERSION_CODES.O)
@Composable
fun ReminderApp(modifier: Modifier = Modifier) {
    val taskViewModel: TaskViewModel =
        viewModel(factory = TaskViewModel.Factory)
    Scaffold {
        Surface(
            modifier = modifier.fillMaxSize()
        ) {
            TaskScreen(taskViewModel)
        }
    }
}