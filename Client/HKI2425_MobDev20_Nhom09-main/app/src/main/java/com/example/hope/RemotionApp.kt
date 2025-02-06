package com.example.hope

import android.app.Application
import android.os.Build
import androidx.annotation.RequiresApi
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AccountCircle
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.MailOutline
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.outlined.CheckCircle
import androidx.compose.material.icons.outlined.FavoriteBorder
import androidx.compose.material.icons.outlined.Menu
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationDrawerItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.example.hope.auth.ui.sign_in.ProfileApp
import com.example.hope.auth.ui.sign_in.UserData
import com.example.hope.other.ui.InfoScreen
import com.example.hope.other.ui.InstructScreen
import com.example.hope.other.ui.SettingScreen
import com.example.hope.mood_tracker.NoteApp
import com.example.hope.navigation.Destinations
import com.example.hope.analyst.ui.AnaScreen
import com.example.hope.analyst.ui.ChartScreen
import com.example.hope.analyst.ui.MultipleChoiceScreen
import com.example.hope.analyst.ui.VoiceScreen
import com.example.hope.dass21.ui.form.ResultScreen
import com.example.hope.dass21.ui.form.StressAnxietyDepressionForm
import com.example.hope.dass21.ui.viewmodel.SurveyViewModel
import com.example.hope.reminder.ReminderApp
import kotlinx.coroutines.launch


data class BottomNavItem(
    val title: String,
    val selectedIcon: ImageVector,
    val unselectedIcon: ImageVector,
    val destination: String
)

@RequiresApi(Build.VERSION_CODES.O)
@Composable
fun RemotionApp(
    modifier: Modifier = Modifier,
    userData: UserData?,
    onSignOut: () -> Unit,
    application: Application
) {
    val navController = rememberNavController()
    val drawerState = rememberDrawerState(DrawerValue.Closed)
    val scope = rememberCoroutineScope()
    val surveyViewModel: SurveyViewModel = viewModel()

    // Để lưu trạng thái mục đã chọn
    var selectedDestination by remember { mutableStateOf(Destinations.NOTE_APP) }
    val currentBackStackEntry by navController.currentBackStackEntryAsState()
    val currentDestination = currentBackStackEntry?.destination?.route

    val items = listOf(
        BottomNavItem(
            title = "Tracker",
            selectedIcon = Icons.Filled.Favorite,
            unselectedIcon = Icons.Outlined.FavoriteBorder,
            destination = Destinations.NOTE_APP
        ),
        BottomNavItem(
            title = "Analyst",
            selectedIcon = Icons.Filled.CheckCircle,
            unselectedIcon = Icons.Outlined.CheckCircle,
            destination = Destinations.ANA_SCREEN
        ),
        BottomNavItem(
            title = "Reminder",
            selectedIcon = Icons.Filled.Menu,
            unselectedIcon = Icons.Outlined.Menu,
            destination = Destinations.REMINDER_APP
        ),
        BottomNavItem(
            title = "Setting",
            selectedIcon = Icons.Filled.Settings,
            unselectedIcon = Icons.Outlined.Settings,
            destination = Destinations.SETTING_SCREEN
        ),
    )
    ModalNavigationDrawer(
        drawerState = drawerState,
        drawerContent = {
            ModalDrawerSheet(modifier = Modifier.width(LocalConfiguration.current.screenWidthDp.dp*3/4)) {
                DrawableContent(
                    selectedDestination = selectedDestination,
                    onDestinationSelected = { destination ->
                        selectedDestination = destination
                        navController.navigate(destination)
                        scope.launch { drawerState.close() }  // Đóng drawer sau khi điều hướng
                    }
                )
            }
        }
    ) {
        Scaffold(
            topBar = {
                TopBar(
                    onOpenDrawer = {
                        scope.launch {
                            drawerState.apply {
                                if (isClosed) open() else close()
                            }
                        }
                    },
                    onClickAccount = {
                        if (selectedDestination != Destinations.PROFILE_APP) {
                            selectedDestination = Destinations.PROFILE_APP
                            navController.navigate(Destinations.PROFILE_APP) {
                                // Đảm bảo không thêm destination mới vào back stack nếu đã có
                                launchSingleTop = true
                            }
                        }
                    }

                )
            },
            bottomBar = {
                NavigationBar {
                    items.forEachIndexed { index, item ->
                        NavigationBarItem(
                            selected = currentDestination == item.destination,
                            onClick = {
                                // Chỉ điều hướng nếu destination khác hiện tại
                                if (selectedDestination != item.destination) {
                                    selectedDestination = item.destination
                                    navController.navigate(item.destination) {
                                        // Đảm bảo không thêm destination mới vào back stack nếu đã có
                                        launchSingleTop = true
                                    }
                                }
                            },
                            alwaysShowLabel = false,
                            label = {
                                Text(text = item.title)
                            },
                            icon = {
                                Icon(
                                    imageVector = if (currentDestination == item.destination) item.selectedIcon
                                    else item.unselectedIcon,
                                    contentDescription = null
                                )
                            }
                        )
                    }
                }
            }

        ) { padding ->
            NavHost(
                navController = navController,
                startDestination = Destinations.NOTE_APP,
                modifier = Modifier.padding(padding)
            ) {
                composable(Destinations.NOTE_APP) {
                    NoteApp()
                }
                composable(Destinations.ANA_SCREEN) {
                    AnaScreen(navController)
                }
                composable(Destinations.REMINDER_APP) {
                    ReminderApp()
                }
                composable(Destinations.PROFILE_APP) {
                    ProfileApp(userData = userData, onSignOut = onSignOut)
                }
                composable(Destinations.INFO_SCREEN) {
                    InfoScreen()
                }
                composable(Destinations.SETTING_SCREEN) {
                    SettingScreen()
                }
                composable(Destinations.INSTRUCT_SCREEN) {
                    InstructScreen()
                }
                composable(Destinations.MPC_SCREEN) {
                    StressAnxietyDepressionForm(viewModel = surveyViewModel,
                        onHomeBack = {},// Trở về màn hình chính
                        onSubmit = {
                            navController.navigate("result")
                        })
                }
                composable(Destinations.VOICE_SCREEN) {
                    VoiceScreen(application)
                }
                composable(Destinations.CHART_SCREEN) {
                    ChartScreen()
                }
//                composable("form") {
//
//                }
                composable("result") {
                    ResultScreen(
                        resultResponse = surveyViewModel.resultResponse.value,
                        onNextStage = { navController.navigate(Destinations.MPC_SCREEN) },
                        onReturnHome = {navController.navigate(Destinations.ANA_SCREEN)}
                    )
                }
            }
        }
    }
}

@Composable
fun DrawableContent(
    selectedDestination: String,
    onDestinationSelected: (String) -> Unit
) {
    Text(
        text = "Remotion",
        fontSize = 24.sp,
        modifier = Modifier.padding(start = 20.dp, top = 20.dp, bottom = 16.dp)
    )
    HorizontalDivider()
    Spacer(modifier = Modifier.height(4.dp).padding(top = 16.dp))

    NavigationDrawerItem(
        icon = {
            Icon(
                imageVector = Icons.Default.Info,
                contentDescription = null,
                modifier = Modifier.padding(start = 10.dp, end = 2.dp).size(25.dp)
            )
        },
        label = {
            Text(text = "About us", fontSize = 20.sp)
        },
        selected = selectedDestination == Destinations.INFO_SCREEN,  // Kiểm tra xem mục này có được chọn không
        onClick = {
            onDestinationSelected(Destinations.INFO_SCREEN)
        }
    )
    Spacer(modifier = Modifier.height(4.dp))

    NavigationDrawerItem(
        icon = {
            Icon(
                imageVector = Icons.Default.MailOutline,
                contentDescription = null,
                modifier = Modifier.padding(start = 10.dp, end = 2.dp).size(25.dp)
            )
        },
        label = {
            Text(text = "How to use", fontSize = 20.sp)
        },
        selected = selectedDestination == Destinations.INSTRUCT_SCREEN,  // Kiểm tra xem mục này có được chọn không
        onClick = {
            onDestinationSelected(Destinations.INSTRUCT_SCREEN)
        }
    )

}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TopBar(
    onOpenDrawer: () -> Unit,
    onClickAccount: () -> Unit
) {
    TopAppBar(
        navigationIcon = {
            Icon(
                imageVector = Icons.Default.Menu,
                contentDescription = null,
                modifier = Modifier
                    .padding(start = 16.dp, end = 4.dp)
                    .size(30.dp)
                    .clickable { onOpenDrawer() }
            )
        },
        title = { Text(text = "Have a nice day") },
        actions = {
            Icon(
                imageVector = Icons.Default.AccountCircle,
                contentDescription = null,
                modifier = Modifier
                    .padding(start = 8.dp, end = 16.dp)
                    .size(30.dp)
                    .clickable {
                        onClickAccount()
                    }
            )
        }
    )
}