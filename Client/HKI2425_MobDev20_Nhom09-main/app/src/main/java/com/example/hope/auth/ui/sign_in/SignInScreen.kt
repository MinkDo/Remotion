package com.example.hope.auth.ui.sign_in

import android.content.Context
import android.net.Uri
import android.util.AttributeSet
import android.widget.FrameLayout
import android.widget.VideoView
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.ViewCompositionStrategy
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.example.hope.R

class FullHeightVideoView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : VideoView(context, attrs, defStyleAttr) {

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val screenHeight = MeasureSpec.getSize(heightMeasureSpec)
        val width = MeasureSpec.getSize(widthMeasureSpec)

        val videoAspectRatio = 9f / 16f // Giả sử video có tỷ lệ 16:9
        val adjustedHeight = screenHeight
        val adjustedWidth = (adjustedHeight * videoAspectRatio).toInt()

        setMeasuredDimension(adjustedWidth, adjustedHeight)
    }
}


@Composable
fun SignInScreen(
    state: SignInState,
    onSignInClick: () -> Unit
) {
    val context = LocalContext.current

    Box(
        modifier = Modifier
            .fillMaxSize()
    ) {
        // Nền video
        AndroidView(
            factory = { context ->
                FullHeightVideoView(context).apply {
                    setVideoURI(Uri.parse("android.resource://${context.packageName}/${R.raw.background_video}"))
                    layoutParams = FrameLayout.LayoutParams(
                        FrameLayout.LayoutParams.MATCH_PARENT,
                        FrameLayout.LayoutParams.MATCH_PARENT
                    )
                    start()
                    setOnPreparedListener { it.isLooping = true } // Loop video
                }
            },
            modifier = Modifier.fillMaxSize()
        )


        // Nút Sign In
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(bottom = 100.dp),
            contentAlignment = Alignment.BottomCenter
        ) {
            Button(
                onClick = onSignInClick,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 28.dp)
                    .height(46.dp)
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.Center
                ) {
                    // Logo Google
                    Image(
                        painter = painterResource(id = R.drawable.googleicon),
                        contentDescription = "Google Logo",
                        modifier = Modifier
                            .size(36.dp)
                            .padding(end = 12.dp),
                    )
                    // Text nút
                    Text(text = "Sign in with Google", fontSize = 16.sp)
                }
            }
        }
    }
}
