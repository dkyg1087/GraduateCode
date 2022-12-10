package com.example.myapplication;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.POST;

public interface ControlAPI {
    @POST("control")

    Call<DataModal> createPost(@Body DataModal dataModal);
}