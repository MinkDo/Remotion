package com.example.hope

import android.app.Application

// file này khởi tạo container, là dữ liệu cần dùng trong cả ứng dụng
class NoteApplication : Application() {
    lateinit var container: AppContainer

    override fun onCreate() {
        super.onCreate()
        container = AppDataContainer(this)
    }
}