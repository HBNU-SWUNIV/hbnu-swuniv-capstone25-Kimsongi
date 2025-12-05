package com.capstone.suhwagi;

import android.util.Base64;

import org.json.JSONException;
import org.json.JSONObject;

import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.util.concurrent.TimeUnit;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;

public class DevTokenIssuer {
    private static final String API_KEY = "devkey";
    private static final String API_SECRET = "secret";

    private DevTokenIssuer() {}

    public static String createToken(String identity, String roomName) {
        long now = TimeUnit.MILLISECONDS.toSeconds(System.currentTimeMillis());
        long ttl = TimeUnit.HOURS.toSeconds(6);

        JSONObject header;
        JSONObject video;
        JSONObject payload;

        try {
            header = new JSONObject()
                .put("alg", "HS256")
                .put("typ", "JWT");

            video = new JSONObject()
                .put("roomCreate", true)
                .put("roomJoin", true)
                .put("room", roomName)
                .put("canPublish", true)
                .put("canSubscribe", true)
                .put("canPublishData", true);

            payload = new JSONObject()
                .put("video", video)
                .put("sub", identity)
                .put("iss", API_KEY)
                .put("nbf", now)
                .put("exp", now + ttl);
        } catch (JSONException e) {
            throw new RuntimeException(e);
        }

        String header64 = base64UrlEncode(header.toString());
        String payload64 = base64UrlEncode(payload.toString());
        String unsignedToken = header64 + "." + payload64;

        byte[] signatureBytes = hmacSha256(unsignedToken);
        String signature = base64UrlEncode(signatureBytes);

        return unsignedToken + "." + signature;
    }

    private static String base64UrlEncode(byte[] input) {
        return Base64.encodeToString(
            input,
            Base64.URL_SAFE | Base64.NO_PADDING | Base64.NO_WRAP
        );
    }

    private static String base64UrlEncode(String input) {
        return base64UrlEncode(input.getBytes());
    }

    private static byte[] hmacSha256(String signingInput) {
        try {
            Mac mac = Mac.getInstance("HmacSHA256");
            SecretKeySpec secret = new SecretKeySpec(API_SECRET.getBytes(), "HmacSHA256");
            mac.init(secret);

            return mac.doFinal(signingInput.getBytes());
        } catch (NoSuchAlgorithmException | InvalidKeyException e) {
            throw new RuntimeException(e);
        }
    }
}