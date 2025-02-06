package com.example.hope.mood_tracker.data.repository

import com.example.hope.auth.ui.sign_in.GoogleAuthUiClient
import com.example.hope.auth.ui.sign_in.UserData
import com.example.hope.mood_tracker.data.database.Note
import kotlinx.coroutines.flow.Flow

interface NoteRepository {
    val googleAuthUiClient: GoogleAuthUiClient
    val userData: UserData
    fun getAllNotes() : Flow<List<Note>>
//    suspend fun getNoteById(id: Int) : Note?
    suspend fun insertNote(note: Note)
    suspend fun updateNote(note: Note)
    suspend fun deleteNote(note: Note)
    suspend fun syncNotesFromFirestore()
    suspend fun syncOfflineQueue()
    suspend fun clearRoomDataForNewUser()
}