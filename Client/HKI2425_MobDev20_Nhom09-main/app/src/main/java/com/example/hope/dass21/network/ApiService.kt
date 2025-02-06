package com.example.hope.dass21.network

import android.security.identity.ResultData
import com.example.hope.dass21.data.model.FormData
import com.example.hope.dass21.data.model.PredictForm
import com.example.hope.dass21.data.model.PredictResponse
import com.example.hope.dass21.data.model.ResultResponse
import retrofit2.Call
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.Headers
import retrofit2.http.POST

interface ApiService {
    @POST("/form")
    suspend fun submitForm(@Body formData: FormData): Response<ResultResponse>

    @POST("/predict")
    suspend fun predictForm(@Body predictForm: PredictForm): Response<PredictResponse>
}