package com.example.hope.mood_tracker.ui

import android.os.Build
import androidx.annotation.RequiresApi
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.ViewModelProvider.AndroidViewModelFactory.Companion.APPLICATION_KEY
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.example.hope.NoteApplication
import com.example.hope.mood_tracker.data.database.Note
import com.example.hope.mood_tracker.data.repository.NoteRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.time.LocalDate

class NoteViewModel(
    private val noteRepository: NoteRepository
): ViewModel() {

    private val _notes = noteRepository.getAllNotes()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(), emptyList())

    @RequiresApi(Build.VERSION_CODES.O)
    private val _state = MutableStateFlow(NoteState())

    @RequiresApi(Build.VERSION_CODES.O)
    val state = combine(_state, _notes) { state, notes ->
        state.copy(
            notes = notes
        )
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), NoteState())

    @RequiresApi(Build.VERSION_CODES.O)
    fun onEvent(event: NoteEvent) {
        when(event) {
            is NoteEvent.SetContent -> {
                _state.update { it.copy(
                    content = event.content
                ) }
            }
            is NoteEvent.SetDate -> {
                _state.update { it.copy(
                    date = event.date
                ) }
            }
            is NoteEvent.SetEmotion -> {
                _state.update { it.copy(
                    emotion = event.emotion
                ) }
            }
            NoteEvent.ShowDialog -> {
                _state.update { it.copy(
                    isAddingNote = true
                ) }
            }
            NoteEvent.HideDialog -> {
                _state.update { it.copy(
                    isAddingNote = false,
                    content = "",
                    emotion = 0
                ) }
            }

            // *lưu vào database
            NoteEvent.SaveNote -> {
                val content = state.value.content
                val date = state.value.date
                val emotion = state.value.emotion
                if(content.isBlank()) {
                    return
                }
                val note = Note(
                    userId = noteRepository.userData.userId,
                    content = content,
                    date = date,
                    emotion = emotion
                )
                viewModelScope.launch {
                    noteRepository.insertNote(note)
                }
                _state.update { it.copy(
                    isAddingNote = false,
                    content = "",
                    emotion = 0
                ) }
            }
            is NoteEvent.DeleteNote -> {
                viewModelScope.launch {
                    noteRepository.deleteNote(event.note)
                }
            }
            is NoteEvent.UpdateNote -> {
                viewModelScope.launch {
                    noteRepository.updateNote(event.note)
                }
            }
        }
    }

    @RequiresApi(Build.VERSION_CODES.O)
    fun getEmotion(date: LocalDate): Int {
        // Kiểm tra nếu ngày đã tồn tại trong danh sách _notes
        return state.value.notes.find { it.date == date }?.emotion ?: 0
    }

    @RequiresApi(Build.VERSION_CODES.O)
    fun getNoteByDate(date: LocalDate) : Note? {
        return state.value.notes.find { it.date == date }
    }

    fun clearRoomDataForNewUser() {
        viewModelScope.launch {
            noteRepository.clearRoomDataForNewUser()
        }
    }

    fun syncOfflineQueue() {
        viewModelScope.launch {
            noteRepository.syncOfflineQueue()
        }
    }


    companion object {
        val Factory: ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val application = (this[APPLICATION_KEY] as NoteApplication)
                val noteRepository = application.container.noteRepository
                NoteViewModel(noteRepository = noteRepository)
            }
        }
    }

}