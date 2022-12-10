package com.example.myapplication;

public class DataModal {

    // string variables for our name and job
    private String action;
    private String email;
    private String control;

    public DataModal(String action, String email,String control) {
        this.action = action;
        this.email = email;
        this.control = control;
    }

    public String getAction() {
        return action;
    }

    public void setAction(String action) {
        this.action = action;
    }

    public String getContent() {
        return email;
    }

    public void setContent(String email) {
        this.email = email;
    }

    public String getControl() {
        return control;
    }

    public void setControl(String control) {
        this.control = control;
    }

}
