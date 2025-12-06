package com.capstone.suhwagi

import io.livekit.android.ConnectOptions
import io.livekit.android.events.EventListenable
import io.livekit.android.room.Room
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.launch

fun <T> EventListenable<T>.collectInScope(
    scope: CoroutineScope,
    action: suspend (value: T) -> Unit
) = scope.launch {
    events.collect { value -> action(value) }
}

fun Room.connectInScope(
    scope: CoroutineScope,
    url: String,
    token: String,
    options: ConnectOptions,
    callback: (Boolean) -> Unit
) = scope.launch {
    try {
        connect(url, token, options)
        localParticipant.setCameraEnabled(true)
        localParticipant.setMicrophoneEnabled(true)

        callback(true)
    } catch (e: Exception) {
        callback(false)
    }
}