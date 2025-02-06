package com.example.hope.mood_tracker.ui

import com.example.hope.mood_tracker.data.database.Note
import java.time.LocalDate

sealed interface NoteEvent {
    object SaveNote: NoteEvent
    data class UpdateNote(val note: Note) : NoteEvent
    data class DeleteNote(val note: Note) : NoteEvent
    data class SetContent(val content: String): NoteEvent
    data class SetDate(val date: LocalDate): NoteEvent
    data class SetEmotion(val emotion: Int): NoteEvent
    object ShowDialog: NoteEvent
    object HideDialog: NoteEvent
}