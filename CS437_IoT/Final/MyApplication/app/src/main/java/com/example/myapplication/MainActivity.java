package com.example.myapplication;

import androidx.appcompat.app.AppCompatActivity;


import android.app.AlertDialog;
import android.content.DialogInterface;
import android.os.Bundle;
import android.text.InputType;
import android.view.View;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;


public class MainActivity extends AppCompatActivity {


    private String m_string;
    private void sendPost(String dir,String action,String content,String control) {
        Retrofit retrofit = new Retrofit.Builder().baseUrl("http://172.16.109.23:5000").addConverterFactory(GsonConverterFactory.create()).build();
        DataModal modal = new DataModal(action, content, control);
        Call<DataModal> call = retrofit.create(EmailAPI.class).createPost(modal);
        if (dir == "email") {
            EmailAPI api = retrofit.create(EmailAPI.class);
            call = api.createPost(modal);
        } else if (dir == "control") {
            ControlAPI api = retrofit.create(ControlAPI.class);
            call = api.createPost(modal);
        }
        call.enqueue(new Callback<DataModal>() {
            @Override
            public void onResponse(Call<DataModal> call, Response<DataModal> response) {
                //Toast.makeText(MainActivity.this,"Success",Toast.LENGTH_SHORT).show();
            }
            @Override
            public void onFailure(Call<DataModal> call, Throwable t) {
                //Toast.makeText(MainActivity.this,"Failed",Toast.LENGTH_SHORT).show();
            }
        });
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        WebView myWebView = (WebView) findViewById(R.id.webView);
        myWebView.setWebViewClient(new WebViewClient());
        myWebView.setInitialScale(156);
        myWebView.loadUrl("http://172.16.109.23:8000");

        Button email = (Button) findViewById(R.id.button);
        email.setOnClickListener(view -> {
            AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
            builder.setTitle("Enter your email:");

            final EditText input = new EditText(MainActivity.this);
            input.setInputType(InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS);
            builder.setView(input);

            builder.setPositiveButton("Set", (dialog, which) -> {
                m_string = input.getText().toString();
                EditText edt = (EditText)findViewById(R.id.editTextTextEmailAddress);
                edt.setText(m_string);
                sendPost("email","set",m_string," ");
            });
            builder.setNegativeButton("Cancel", (dialog, which) -> dialog.cancel());

            builder.show();
        }

        );
        Button delEmail = (Button) findViewById(R.id.button2);
        delEmail.setOnClickListener(view -> {
            sendPost("email","del"," "," ");
            EditText edt = (EditText)findViewById(R.id.editTextTextEmailAddress);
            edt.setText("");
            m_string = "";
                }

        );
        Button left = (Button) findViewById(R.id.button4);
        left.setOnClickListener(view->{
            sendPost("control","left"," ","left");
        });
        Button right = (Button) findViewById(R.id.button5);
        right.setOnClickListener(view->{
            sendPost("control","right"," ","right");
        });
    }

}