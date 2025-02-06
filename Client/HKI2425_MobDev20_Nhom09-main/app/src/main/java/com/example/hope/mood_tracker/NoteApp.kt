package com.example.hope.mood_tracker

import android.annotation.SuppressLint
import android.os.Build
import androidx.annotation.RequiresApi
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import com.example.hope.mood_tracker.ui.NoteScreen
import com.example.hope.mood_tracker.ui.NoteViewModel
import androidx.lifecycle.viewmodel.compose.viewModel

@SuppressLint("UnusedMaterial3ScaffoldPaddingParameter")
@RequiresApi(Build.VERSION_CODES.O)
@Composable
fun NoteApp(modifier: Modifier = Modifier) {
    Scaffold {
        Surface(
            modifier = modifier.fillMaxSize()
        ) {
            val noteViewModel: NoteViewModel =
                viewModel(factory = NoteViewModel.Factory)
            val state by noteViewModel.state.collectAsState()
//            NoteScreen(state = state, onEvent = noteViewModel::onEvent, noteViewModel::check)
            NoteScreen(noteViewModel = noteViewModel)
        }
    }
}