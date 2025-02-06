package com.example.hope.mood_tracker.data.repository

import android.content.Context
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import com.example.hope.auth.ui.sign_in.GoogleAuthUiClient
import com.example.hope.auth.ui.sign_in.UserData
import com.example.hope.mood_tracker.data.database.Note
import com.example.hope.mood_tracker.data.database.NoteDao
import com.example.hope.mood_tracker.utils.NetworkUtils
import com.google.firebase.firestore.FirebaseFirestore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.tasks.await
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.util.concurrent.ConcurrentLinkedQueue

class NoteRepositoryImpl(
    private val noteDao: NoteDao,
    override val googleAuthUiClient: GoogleAuthUiClient,
    private val context: Context
) : NoteRepository {

    private val offlineQueue: ConcurrentLinkedQueue<OfflineAction> = ConcurrentLinkedQueue()

    // Định nghĩa OfflineAction để lưu thông tin thao tác
    data class OfflineAction(
        val actionType: String, // "ADD", "UPDATE", "DELETE"
        val note: Note
    )

    // chỗ này cập nhật so với code cũ, important nhá
    override val userData: UserData
        get() = googleAuthUiClient.getCurrentUser()


    private val firestore = FirebaseFirestore.getInstance()

    override fun getAllNotes(): Flow<List<Note>> {
        Log.d("UserData trong Repo", "${userData}")
        return noteDao.getNotes(userId = userData.userId)
    }

    override suspend fun insertNote(note: Note) {
        // Lưu ghi chú vào Room
        noteDao.insertNote(note)

        // Kiểm tra kết nối internet và đồng bộ lên Firestore nếu có kết nối
        if (NetworkUtils.isNetworkAvailable(context = context)) {
            addNoteFirestore(note) // Sử dụng bản ghi đã có id hợp lệ
        } else {
            offlineQueue.add(OfflineAction("ADD", note))
        }
    }

    override suspend fun updateNote(note: Note) {
        noteDao.updateNote(note)
        Log.d("update func", "nó có vào không?")

        if (NetworkUtils.isNetworkAvailable(context = context)) {
            updateNoteFirestore(note)
        } else {
            offlineQueue.add(OfflineAction("UPDATE", note))
        }
    }

    override suspend fun deleteNote(note: Note) {
        noteDao.deleteNote(note)
        // phải thêm logic là xóa note nào nhé
        if (NetworkUtils.isNetworkAvailable(context = context)) {
            deleteNoteFirestore(note)
        } else {
            offlineQueue.add(OfflineAction("DELETE", note))
        }
    }

    // Đồng bộ ghi chú từ Room lên Firestore, cái này chỉ được cho mỗi tạo và update thôi
    private suspend fun addNoteFirestore(note: Note) {
        try {
            val noteMap = mapOf(
                "userId" to note.userId,
                "date" to note.date.toString(),
                "content" to note.content,
                "emotion" to note.emotion
            )
            firestore.collection("notes")
                .document()
                .set(noteMap)
                .await() // Sử dụng await để chờ Firebase trả về kết quả
        } catch (e: Exception) {
            // Xử lý lỗi nếu có
            e.printStackTrace()
        }
    }

    private suspend fun deleteNoteFirestore(note: Note) {
        val dateStr = note.date.toString() // phải chuyển sang String cho bên Firebase
        try {
            firestore.collection("notes")
                .whereEqualTo("userId", note.userId)
                .whereEqualTo("date", dateStr)
                .get()
                .addOnSuccessListener { documents ->
                    if (!documents.isEmpty) {
                        // Xóa document đầu tiên tìm thấy, vì mỗi ngày chỉ có 1 note của user
                        val document = documents.documents[0]
                        firestore.collection("notes").document(document.id).delete()
                        Log.e("Firestore", "Xóa rồi")
                    }
                }
                .addOnFailureListener { exception ->
                    // Xử lý lỗi khi truy vấn hoặc xóa
                    Log.e("Firestore", "Error deleting document: ${exception.message}")
                }
                .await()
        }
        catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private suspend fun updateNoteFirestore(note: Note) {
        firestore.collection("notes")
            .whereEqualTo("userId", note.userId)
            .whereEqualTo("date", note.date.toString())
            .get()
            .addOnSuccessListener { documents ->
                if (!documents.isEmpty) {
                    // Lấy document đầu tiên tìm thấy
                    val document = documents.documents[0]
                    firestore.collection("notes")
                        .document(document.id)
                        .update(
                            mapOf(
                                "content" to note.content,
                                "emotion" to note.emotion,
                                "date" to note.date.toString(),
                                "userId" to note.userId
                            )
                        )
                        .addOnSuccessListener {
                            Log.d("Firestore", "Document successfully updated")
                        }
                        .addOnFailureListener { exception ->
                            Log.e("Firestore", "Error updating document: ${exception.message}")
                        }
                } else {
                    Log.e("Firestore", "No document found for the given date and userId")
                }
            }
            .addOnFailureListener { exception ->
                Log.e("Firestore", "Error finding document: ${exception.message}")
            }
            .await()
    }

    // Đồng bộ ghi chú từ Firestore về Room (chỉ lấy ghi chú của userId hiện tại)
    @RequiresApi(Build.VERSION_CODES.O)
    override suspend fun syncNotesFromFirestore() {
        try {
            val querySnapshot = firestore.collection("notes")
                .whereEqualTo("userId", userData.userId)
                .get()
                .await()
            // Lấy danh sách ghi chú từ Firestore mà không bao gồm `id`
            val notes = querySnapshot.documents.mapNotNull { document ->
                val userId = document.getString("userId") ?: return@mapNotNull null
                val dateStr  = document.getString("date") ?: return@mapNotNull null
                val content = document.getString("content") ?: ""
                val emotion = document.getLong("emotion")?.toInt() ?: 0

                val date = LocalDate.parse(dateStr, DateTimeFormatter.ofPattern("yyyy-MM-dd"))
                Log.d("FirestoreData", "Document ID: ${document.id}, userId: $userId, date: $date, content: $content, emotion: $emotion")
                // Tạo một `Note` mới mà không có `id`
                Note(
                    userId = userId,
                    date = date,
                    content = content,
                    emotion = emotion
                )
            }
            Log.d("SyncFunction", "Số lượng notes đã tạo từ Firestore: ${notes.size}")
            // Thêm ghi chú vào Room
            for (note in notes) {
                noteDao.insertNote(note)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    // queue này lưu lại các hành động
    override suspend fun syncOfflineQueue() {
        while (offlineQueue.isNotEmpty()) {
            val action = offlineQueue.poll() // Lấy và xóa phần tử đầu tiên trong hàng đợi

            action?.let {
                when (it.actionType) {
                    "ADD" -> addNoteFirestore(it.note)
                    "UPDATE" -> updateNoteFirestore(it.note)
                    "DELETE" -> deleteNoteFirestore(it.note)
                    else -> {}
                }
            }
        }
        Log.d("SyncOff", "Hoạt động")
    }

    @RequiresApi(Build.VERSION_CODES.O)
    override suspend fun clearRoomDataForNewUser() {
        Log.d("ClearFunc", "Nó có hoạt động không?")
        noteDao.deleteAllNotes()  // Xóa tất cả ghi chú trong Room
        syncNotesFromFirestore()  // Đồng bộ lại dữ liệu từ Firestore
    }
}