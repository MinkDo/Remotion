package com.example.hope.dass21.ui.viewmodel

import android.util.Log
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.hope.dass21.data.model.FormData
import com.example.hope.dass21.data.model.PredictForm
import com.example.hope.dass21.data.model.PredictResponse
import com.example.hope.dass21.data.model.ResultResponse
import com.example.hope.dass21.network.RetrofitClient
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

import kotlinx.coroutines.launch

class SurveyViewModel : ViewModel() {
    val resultResponse = MutableLiveData<ResultResponse?>()
    val errorMessage = MutableLiveData<String?>()
//    val predictResultResponse = MutableLiveData<PredictResponse?>()

    private val _predictResultResponse = MutableStateFlow<PredictResponse?>(null)
    val predictResultResponse = _predictResultResponse.asStateFlow()

    // Gửi dữ liệu form tới API
    fun submitForm(formData: FormData) {
        Log.d("SurveyViewModel", "Form data being sent: $formData")

        viewModelScope.launch {
            try {
                // Gửi yêu cầu API sử dụng Coroutines
                val response = RetrofitClient.apiService.submitForm(formData)

                // Kiểm tra kết quả trả về
                if (response.isSuccessful) {
                    resultResponse.value = response.body()
                    Log.d("SurveyViewModel", "API Response: ${response.body()}")
                } else {
                    errorMessage.value = "Error: ${response.message()}"
                    Log.e("SurveyViewModel", "Error: ${response.message()}")
                }
            } catch (e: Exception) {
                // Xử lý lỗi mạng hoặc lỗi bất thường
                errorMessage.value = "Network error: ${e.message}"
                Log.e("SurveyViewModel", "Network error: ${e.message}")
            }
        }
    }

    fun predictForm(predictForm: PredictForm) {
        Log.d("SurveyViewModel", "Text being sent: $predictForm")

        viewModelScope.launch {
            try {
                // Gửi yêu cầu API sử dụng Coroutines
                val response = RetrofitClient.apiService.predictForm(predictForm)

                // Kiểm tra kết quả trả về
                if (response.isSuccessful) {
                    _predictResultResponse.value  = response.body()
                    Log.d("SurveyViewModel", "API Response: ${response.body()}")
                } else {
                    errorMessage.value = "Error: ${response.message()}"
                    Log.e("SurveyViewModel", "Error: ${response.message()}")
                }
            } catch (e: Exception) {
                // Xử lý lỗi mạng hoặc lỗi bất thường
                errorMessage.value = "Network error: ${e.message}"
                Log.e("SurveyViewModel", "Network error: ${e.message}")
            }
        }
    }
}