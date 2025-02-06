package com.example.hope.analyst.ui

import androidx.compose.animation.core.LinearOutSlowInEasing
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.example.hope.theme.Pink80
import com.example.hope.theme.Purple40
import com.example.hope.theme.Purple80
import com.example.hope.theme.PurpleGrey80
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import com.example.hope.R

@Composable
fun ChartScreen() {
    val mapdata: Map<Int, Int> = mapOf(
        0 to 25,
        1 to 30,
        2 to 40,
        3 to 10,
        4 to 2
    )
    val emotionList: Map<Int, Int> = mapOf(
        0 to R.drawable.happy,
        1 to R.drawable.relax,
        2 to R.drawable.bored,
        3 to R.drawable.sad,
        4 to R.drawable.angry
    )
    PieChart(mapdata)
}

@Composable
fun PieChart(
    data: Map<Int, Int>,
    radiusOuter: Dp = 90.dp,
    chartBarWidth: Dp = 20.dp,
    animDuration: Int = 1000,
) {
    val totalSum = data.values.sum()
    val floatValue = mutableListOf<Float>()

    data.values.forEachIndexed { index, values ->

        floatValue.add(index, 360*values.toFloat() / totalSum)
    }

    val colors = listOf(
//        Color(#949e6a),
        Color(0xFF949E6A),
        Color(0xFFc0c79b),
        Color(0xFFf5d894),
        Color(0XFFdd9e75),
        Color(0XFFc75a55)
    )

    var animationPlayed by remember { mutableStateOf(false) }

    var lastValue = 0f

    val animateSize by animateFloatAsState(
        targetValue = if (animationPlayed) radiusOuter.value * 2f else 0f,
        animationSpec = tween(
            durationMillis = animDuration,
            delayMillis = 0,
            easing = LinearOutSlowInEasing
        )
    )

    val animateRotation by animateFloatAsState(
        targetValue = if (animationPlayed) 90f * 11f else 0f,
        animationSpec = tween(
            durationMillis = animDuration,
            delayMillis = 0,
            easing = LinearOutSlowInEasing
        )
    )

    LaunchedEffect(key1 = true) {
        animationPlayed = true
    }

    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(Modifier.height(120.dp))
        Box(
            modifier = Modifier.size(animateSize.dp),
            contentAlignment = Alignment.Center
            ) {
                Canvas(
                    modifier = Modifier.size(radiusOuter * 2f).rotate(animateRotation),
                ) {
                    floatValue.forEachIndexed { index, value ->
                        drawArc(
                            color = colors[index],
                            lastValue,
                            value,
                            useCenter = false,
                            style = Stroke(chartBarWidth.toPx(), cap = StrokeCap.Butt)
                        )
                        lastValue += value
                    }
                }
        }
    }
}




