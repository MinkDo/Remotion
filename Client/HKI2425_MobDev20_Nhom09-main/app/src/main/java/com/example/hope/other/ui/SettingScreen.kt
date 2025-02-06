package com.example.hope.other.ui

import android.content.Context
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
fun SettingScreen() {
    val context = LocalContext.current
    val sharedPreferences = context.getSharedPreferences("app_preferences", Context.MODE_PRIVATE)

    // Đọc các giá trị từ SharedPreferences
    var isNotificationEnabled by remember { mutableStateOf(sharedPreferences.getBoolean("is_notification_enabled", false)) }
    var snoozeTime by remember { mutableStateOf(sharedPreferences.getInt("snooze_time", 10)) } // Mặc định 10 phút

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Cài đặt bật/tắt thông báo
        Row(
            verticalAlignment = Alignment.CenterVertically, // Căn giữa các phần tử theo chiều dọc
            modifier = Modifier.fillMaxWidth().padding(8.dp) // Căn chiều rộng và thêm khoảng cách xung quanh
        ) {
            // Tiêu đề "Enable Alarm Screen"
            Text(
                text = "Hiển thị thông báo",
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier
                    .weight(1f) // Phần này chiếm không gian còn lại, đẩy Switch sang phải
                    .alpha(if (isNotificationEnabled) 1f else 0.3f) // Điều chỉnh độ mờ của chữ
            )


            // Switch để bật/tắt thông báo
            Switch(
                checked = isNotificationEnabled,
                onCheckedChange = { newValue ->
                    isNotificationEnabled = newValue
                    // Lưu lại trạng thái notification vào SharedPreferences
                    val editor = sharedPreferences.edit()
                    editor.putBoolean("is_notification_enabled", newValue)
                    editor.apply()
                },
                modifier = Modifier.padding(start = 8.dp) // Thêm khoảng cách giữa Text và Switch
            )
        }

        // Cài đặt thời gian báo lại
        Row(
            verticalAlignment = Alignment.CenterVertically, // Đảm bảo các phần tử nằm giữa chiều dọc
            modifier = Modifier.fillMaxWidth().padding(8.dp) // Đảm bảo Row chiếm toàn bộ chiều rộng và có khoảng cách xung quanh
        ) {
            // Tiêu đề "Snooze Time"
            Text(
                text = "Báo lại",
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.weight(1f) // Phần này chiếm không gian còn lại
            )

            TextField(
                value = if (snoozeTime > 0) snoozeTime.toString() else "", // Nếu giá trị là 0 thì để trống
                onValueChange = { newValue ->
                    // Nếu giá trị mới là rỗng, không làm gì cả, chỉ để trống
                    if (newValue.isEmpty()) {
                        snoozeTime = 0 // Hoặc có thể để lại giá trị mặc định là 0
                    } else {
                        // Kiểm tra và đảm bảo giá trị nhập vào là một số hợp lệ
                        val time = newValue.toIntOrNull()
                        if (time != null && time >= 0) {
                            snoozeTime = time
                            // Lưu lại thời gian báo lại vào SharedPreferences nếu có giá trị hợp lệ
                            val editor = sharedPreferences.edit()
                            editor.putInt("snooze_time", snoozeTime)
                            editor.apply()
                        }
                    }
                },
                textStyle = androidx.compose.ui.text.TextStyle(color = Color.Black, fontSize = 16.sp),
                keyboardOptions = KeyboardOptions.Default.copy(keyboardType = KeyboardType.Number), // Đảm bảo bàn phím chỉ nhận số
                modifier = Modifier
                    .width(80.dp) // Cài đặt độ rộng hợp lý cho TextField
                    .padding(horizontal = 8.dp) // Thêm khoảng cách bên trái và phải của TextField
            )

            // Text "minute"
            Text(
                text = "phút",
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(start = 8.dp) // Thêm khoảng cách giữa TextField và "minute"
            )
        }


        Spacer(modifier = Modifier.height(16.dp))
    }
}
