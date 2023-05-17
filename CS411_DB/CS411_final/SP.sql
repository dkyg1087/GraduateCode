CREATE PROCEDURE regAvgView()
    BEGIN
        DECLARE done int default 0;
        DECLARE currreg VARCHAR(64);
        DECLARE regcur CURSOR FOR SELECT DISTINCT Region FROM Videos;
        DECLARE CONTINUE HANDLER FOR NOT FOUND SET done=1;

        DROP TABLE IF EXISTS regAvgView;

        CREATE TABLE regAvgView(region VARCHAR(64),avgView BIGINT,totalView BIGINT);
        OPEN regcur;
        REPEAT
            FETCH regcur INTO currreg;
            INSERT INTO regAvgView
            (SELECT Region,AVG(Views),SUM(Views) FROM Videos WHERE Region = currreg GROUP BY Region);
            UNTIL done
            END REPEAT;
            close regcur;
    END//