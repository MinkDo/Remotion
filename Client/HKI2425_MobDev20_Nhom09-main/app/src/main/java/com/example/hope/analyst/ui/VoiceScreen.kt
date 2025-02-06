package com.example.hope.analyst.ui

import android.Manifest
import android.app.Activity
import android.app.Application
import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.SpeechRecognizer.ERROR_NO_MATCH
import android.widget.Toast
import androidx.compose.animation.AnimatedContent
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Mic
import androidx.compose.material.icons.rounded.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.PermissionChecker
import com.example.hope.dass21.data.model.PredictForm
import com.example.hope.dass21.ui.viewmodel.SurveyViewModel

@Composable
fun VoiceScreen(app: Application) {
    var speechText by remember { mutableStateOf("") }
    var isListening by remember { mutableStateOf(false) }
    val speechRecognizer = remember { SpeechRecognizer.createSpeechRecognizer(app) }
    val context = LocalContext.current
    val surveyViewModel = SurveyViewModel()
    val predictResultResponse = surveyViewModel.predictResultResponse.collectAsState().value

    var displayResult by remember { mutableStateOf<String?>(null) }
    // Bắt đầu nhận diện giọng nói
    fun startListening() {
        if (ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.RECORD_AUDIO
            ) != PermissionChecker.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                context as Activity,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                0
            )
        } else {
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            }
            speechRecognizer.startListening(intent)
            isListening = true
        }
    }

    // Dừng nhận diện giọng nói
    fun stopListening() {
        speechRecognizer.stopListening()
        isListening = false
    }

    // Lắng nghe các sự kiện từ SpeechRecognizer
    LaunchedEffect(speechRecognizer) {
        speechRecognizer.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(p0: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(p0: Float) {}
            override fun onBufferReceived(p0: ByteArray?) {}
            override fun onEndOfSpeech() {}

            override fun onError(error: Int) {
                if (error == ERROR_NO_MATCH) {
                    Toast.makeText(context, "No speech match", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onResults(results: Bundle?) {
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                speechText += matches?.get(0) ?: ""
            }

            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })
    }


    Scaffold(
        floatingActionButton = {
            FloatingActionButton(
                onClick = {
                    if (isListening) {
                        stopListening()
                    } else {
                        startListening()
                    }
                }
            ) {
                AnimatedContent(targetState = isListening) { isListening ->
                    if (isListening) {
                        Icon(imageVector = Icons.Rounded.Stop, contentDescription = null)
                    } else {
                        Icon(imageVector = Icons.Rounded.Mic, contentDescription = null)
                    }
                }
            }
        }
    ) { padding ->

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(top = 200.dp, start = 16.dp, end = 16.dp)
        ) {
            Text(
                text = "Phân tích cảm xúc",
                style = TextStyle(
                    fontSize = 24.sp,   // Điều chỉnh kích thước chữ
                    fontWeight = FontWeight.Bold  // Đặt chữ đậm
                ),
                modifier = Modifier.padding(bottom = 4.dp)
            )

            TextField(
                value = speechText,
                onValueChange = { speechText = it },
                placeholder = { Text("Nhập văn bản") },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 16.dp)
            )

            Row {
                Button(
                    onClick = {
                        speechText = ""
                        displayResult = ""
                    },
                    modifier = Modifier
                        .weight(1f) // Chia đều không gian
                        .fillMaxWidth()
                ) {
                    Text("Xóa")
                }

                Spacer(modifier = Modifier.width(8.dp))

                val predictForm = PredictForm(
                    speechText
                )
                Button(
                    onClick = {
                        surveyViewModel.predictForm(predictForm)
                    },
                    modifier = Modifier
                        .weight(1f) // Chia đều không gian
                        .fillMaxWidth()
                ) {
                    Text("Gửi")
                }
            }
            LaunchedEffect(predictResultResponse) {
                predictResultResponse?.predicted_emotion?.let { displayResult = it }
            }

            // Hiển thị kết quả từ trạng thái riêng
            displayResult?.let {
                Text(
                    text = it,
                    modifier = Modifier.padding(top = 16.dp),
                    style = TextStyle(
                        fontSize = 18.sp,   // Điều chỉnh kích thước chữ
                        fontWeight = FontWeight.Bold  // Đặt chữ đậm
                    )
                )
            }
        }
    }
}