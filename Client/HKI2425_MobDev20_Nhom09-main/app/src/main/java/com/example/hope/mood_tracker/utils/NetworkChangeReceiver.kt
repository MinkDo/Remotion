package com.example.hope.mood_tracker.utils

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import kotlinx.coroutines.DelicateCoroutinesApi
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch

class NetworkChangeReceiver(
    private val syncOfflineQueue: () -> Unit
) : BroadcastReceiver() {
    @OptIn(DelicateCoroutinesApi::class)
    override fun onReceive(context: Context, intent: Intent) {
        if (NetworkUtils.isNetworkAvailable(context)) {
            // Khởi chạy coroutine để gọi hàm suspend
            GlobalScope.launch {
                syncOfflineQueue()
            }
        }
    }
}