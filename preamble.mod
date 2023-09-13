MODULE ExperimentalWall
    LOCAL PERS arcbeaddata exp_bead_data:=[3.367,[2,0.1,0,0],[329,0,0.9,43.009,0,0,0,0,0],135];
    LOCAL PERS arcstartenddata exp_start_end:=[[2,0.5,0,0,[0,0,1.0,75,0,0,0,0,0],6.77333,7,[0,0,1.0,75,0,0,0,0,0]],[0,0,[0,0,1.0,75,0,0,0,0,0],0.02,[0,0,0.5,75,0,0,0,0,0],0,0.5]];
    LOCAL PERS arctrkdata exp_track_data:=[5,0,0,0,0,40];

    PROC ExperimentBuild()

        VAR robtarget dest:=[[0, 0, 0], [0.000260569,-0.674916,-0.737894,0.00491837], [-1,-1,-3,0], [9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];

        tPrintGun:=OffsToolXYZ(tWeldGun,[0,0,0]);
        tSearchTool:=tSpotSense;

        SetOriPoint;

CodeGoesHere
    ENDPROC

    ! Routine for setting the origin
    LOCAL PROC SetOriPoint(\switch dummy)
 
        ! Defining variables
        VAR num answer;
        VAR robtarget point;
        VAR robtarget zeroPoint;
        VAR pose searchResult;
        VAR btnres location;
        VAR errnum maxTime;
        VAR num MAX_WOBJ_CHANGE:=500;

        ! UI for setting work object
        UIMsgBox\Header:="Set Work Object Origin",
        "To Set/Modify the wobj, Press OK, otherwise, Press Cancel."
        \MsgLine2:="After OK, jog the robot to be centered over the desired wobj origin."
        \MsgLine3:="The wire should be placed within 1 inch from the baseplate."
        \MsgLine4:="Press play once the robot is in the desired position."
        \MsgLine5:="WARNING: Pay attention to robot movement if pressing Cancel."
        \Buttons:=btnOKCancel
        \Icon:=iconInfo
        \Result:=answer;
        IF answer=resOK THEN
            !Zeroing Out Workobjects
            obParamWall:=[FALSE,TRUE,"",[[0,0,0],[1,0,0,0]],[[0,0,0],[1,0,0,0]]];
            WobjWorldAlign obParamWall;
        ENDIF
         
        ! If "Cancel" is selected, the work object is left how it is
        IF answer=resCancel THEN
            RETURN;
        ENDIF
            
        Stop;

        !Operator Robot Movement
        WaitTime\inpos,0.1;
            
        ! Capture Current Translation from selected wobj origin using current location of TCP
        ! and set as origin
        point:=CRobT(\tool:=tWeldGun,\wobj:=obParamWall);
        oriPoint:=point;
            
        WaitTime\inpos,0.1;
            
        obParamWall.oframe.trans:=obParamWall.oframe.trans+point.trans;

        ! Using tSpotSense to determine z height of work object
        zeroPoint:=point;
        zeroPoint.trans:=[0,0,0];
        MoveL RelTool(zeroPoint,0,0,-100),v100,fine,tWeldGun\wobj:=obParamWall;
        MoveJ RelTool(zeroPoint,0,0,-100),v100,fine,tSpotSense\wobj:=obParamWall;
        WaitTime\inpos,0.1;
        point:=CRobT(\tool:=tSpotSense\wobj:=obParamWall);
        WaitTime\inpos,0.1;
        zeroPoint.robconf:=point.robconf;
        Search_1D\Laser,searchResult,RelTool(zeroPoint,0,0,-25),zeroPoint,v100,tSpotSense\wobj:=obParamWall;
        obParamWall.oframe.trans.z:=obParamWall.oframe.trans.z+searchResult.trans.z;

        PDispOff;
           
    ENDPROC

ENDMODULE