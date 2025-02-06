package com.example.hope

import android.content.Context
import android.content.IntentFilter
import android.content.SharedPreferences
import android.net.ConnectivityManager
import android.os.Build
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.IntentSenderRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.annotation.RequiresApi
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.hope.auth.ui.sign_in.GoogleAuthUiClient
import com.example.hope.auth.ui.sign_in.SignInScreen
import com.example.hope.auth.ui.sign_in.SignInViewModel
import com.example.hope.mood_tracker.data.repository.NoteRepository
import com.example.hope.mood_tracker.data.repository.NoteRepositoryImpl
import com.example.hope.mood_tracker.ui.NoteViewModel
import com.example.hope.mood_tracker.ui.NoteViewModel.Companion.Factory
import com.example.hope.mood_tracker.utils.NetworkChangeReceiver
import com.example.hope.theme.HopeTheme
import com.google.android.gms.auth.api.identity.Identity
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {

    private val googleAuthUiClient by lazy {
        GoogleAuthUiClient(
            context = applicationContext,
            oneTapClient = Identity.getSignInClient(applicationContext)
        )
    }

    private lateinit var networkChangeReceiver: NetworkChangeReceiver
    private val noteViewModel: NoteViewModel by viewModels(factoryProducer = { Factory })

    private lateinit var sharedPreferences: SharedPreferences

    @RequiresApi(Build.VERSION_CODES.O)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        installSplashScreen()
        enableEdgeToEdge()

        // khởi tạo biến lateinit
        sharedPreferences = getSharedPreferences("app_preferences", Context.MODE_PRIVATE)

        // chỗ này để gọi hàm đồng bộ khi có mạng
        if (googleAuthUiClient.getSignedInUser() != null) {
            networkChangeReceiver = NetworkChangeReceiver { noteViewModel.syncOfflineQueue() }
            val intentFilter = IntentFilter(ConnectivityManager.CONNECTIVITY_ACTION)
            registerReceiver(networkChangeReceiver, intentFilter)
        }

        setContent {
            HopeTheme {
                val navController = rememberNavController()
                val startDestination = if (googleAuthUiClient.getSignedInUser() != null) {
                    "remotion_app" // Đã đăng nhập, bắt đầu từ RemotionApp
                } else {
                    "sign_in" // Chưa đăng nhập, bắt đầu từ SignIn
                }
                NavHost(navController = navController, startDestination = startDestination) {
                    composable("sign_in") {
                        val viewModel = viewModel<SignInViewModel>()
                        val state by viewModel.state.collectAsStateWithLifecycle()

                        LaunchedEffect(key1 = Unit) {
                            if(googleAuthUiClient.getSignedInUser() != null) {
                                navController.navigate("remotion_app")
                            }
                        }

                        val launcher = rememberLauncherForActivityResult(
                            contract = ActivityResultContracts.StartIntentSenderForResult(),
                            onResult = { result ->
                                if(result.resultCode == RESULT_OK) {
                                    lifecycleScope.launch {
                                        val signInResult = googleAuthUiClient.signInWithIntent(
                                            intent = result.data ?: return@launch
                                        )
                                        viewModel.onSignInResult(signInResult)
                                    }
                                }
                            }
                        )

                        // nó
                        LaunchedEffect(key1 = state.isSignInSuccessfull) {
                            if(state.isSignInSuccessfull) {
                                Toast.makeText(
                                    applicationContext,
                                    "Sign in successfull",
                                    Toast.LENGTH_LONG
                                ).show()
                                noteViewModel.clearRoomDataForNewUser()
                                navController.navigate("remotion_app") {
                                    popUpTo("sign_in") { inclusive = true }
                                }
                                viewModel.resetState()
                                navController.navigate("remotion_app") {
                                    launchSingleTop = true // Tránh điều hướng trùng lặp
                                }
                            }
                        }

                        SignInScreen(
                            state = state,
                            onSignInClick = {
                                lifecycleScope.launch {
                                    val signInIntentSender = googleAuthUiClient.signIn()
                                    launcher.launch(
                                        IntentSenderRequest.Builder(
                                            signInIntentSender ?: return@launch
                                        ).build()
                                    )
                                }
                            }
                        )
                    }
                    composable(route = "remotion_app") {
                        RemotionApp(userData = googleAuthUiClient.getSignedInUser(),
                            onSignOut = {
                                lifecycleScope.launch {
                                    googleAuthUiClient.signOut()
                                    Toast.makeText(
                                        applicationContext,
                                        "Signed out",
                                        Toast.LENGTH_LONG
                                    ).show()
                                    navController.navigate("sign_in") {
                                        popUpTo("remotion_app") { inclusive = true }
                                    }
                                }
                            },
                            application = application
                        )
                    }
                }
            }
        }
    }
    override fun onDestroy() {
        super.onDestroy()
        // Hủy đăng ký BroadcastReceiver khi Activity bị hủy
        unregisterReceiver(networkChangeReceiver)
    }

}
