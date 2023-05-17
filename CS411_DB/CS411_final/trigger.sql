CREATE TRIGGER logVid
    AFTER DELETE ON Videos
    FOR EACH ROW
    BEGIN
        INSERT INTO Logs(VideoTitle,Operation)
        VALUES(old.VideoTitle,"DELETE")
    END//

CREATE TRIGGER logVidu
    AFTER UPDATE ON Videos
    FOR EACH ROW
    BEGIN
        INSERT INTO Logs(VideoTitle,Operation)
        VALUES(old.VideoTitle,"UPDATE");
    END//