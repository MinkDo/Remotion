package com.example.hope

import android.content.Context
import com.example.hope.auth.ui.sign_in.GoogleAuthUiClient
import com.example.hope.database.AppDatabase
import com.example.hope.mood_tracker.data.repository.NoteRepository
import com.example.hope.mood_tracker.data.repository.NoteRepositoryImpl
import com.example.hope.reminder.data.repository.TaskRepository
import com.example.hope.reminder.data.repository.TaskRepositoryImpl
import com.google.android.gms.auth.api.identity.Identity

//cung cấp một nơi duy nhất để quản lý và
// khởi tạo các đối tượng cần thiết cho ứng dụng (như repository và database)
interface AppContainer {
    val noteRepository: NoteRepository
    val taskRepository: TaskRepository
    val googleAuthUiClient: GoogleAuthUiClient
}

class AppDataContainer(private val context: Context): AppContainer {

    override val googleAuthUiClient: GoogleAuthUiClient by lazy {
        GoogleAuthUiClient(
            context = context,
            oneTapClient = Identity.getSignInClient(context)
        )
    }

    override val noteRepository: NoteRepository by lazy {
        NoteRepositoryImpl(AppDatabase.getDatabase(context).noteDao, googleAuthUiClient, context)
    }

    override val taskRepository: TaskRepository by lazy {
        TaskRepositoryImpl(AppDatabase.getDatabase(context).taskDao, context)
    }
}