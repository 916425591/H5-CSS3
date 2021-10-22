/**
 * Created by hzy on 2018/3/13 0013.
 */
var OpenT3DPage = {};
OpenT3DPage.Key1 = "ijianlefanilfabgklgiegnhconldomi";//本地打包的扩展
OpenT3DPage.Key2 = "iboiddmcjdfpoikpjkpimgfaeecbigpb";//web store打包的扩展
OpenT3DPage.MESSAGE_TO_TOP_SHOW_ALERT             = "MESSAGE_TO_TOP_SHOW_ALERT";
OpenT3DPage.ShowingAlertDlg = false;
OpenT3DPage.Lan = navigator.language;
OpenT3DPage.LanMap = {
  "zh-CN": {
    "10001": "通知",
    "10002": "确定"
  },
  "en-US": {
    "10001": "Notify",
    "10002": "OK"
  }
};

OpenT3DPage.GetLan = function(key){
  var obj = this.LanMap[this.Lan];
  if(obj)
    return obj[key];
  else
    return this.LanMap["zh-CN"][key];
}

OpenT3DPage.GetContent = function(url, scallback, fcallback){
  var xmlHttp = new XMLHttpRequest();
  xmlHttp.open("GET", url, true);
  xmlHttp.onreadystatechange=function(){
    if (xmlHttp.readyState==4)
    {
      if (xmlHttp.status==200)
      {
        scallback(xmlHttp.responseText);
      }
      else
      {
        fcallback();
      }
    }
    else{}
  };
  xmlHttp.send(null);
}


OpenT3DPage.PrivateOpen = function(url, target, key){
  window.top.open("chrome-extension://" + key + "/container.html#url=" + encodeURIComponent(url), target);
}

OpenT3DPage.OpenNormalPage = function(url, target){
  window.open(url, target);
}

OpenT3DPage.OpenT3DURL = function(url, target){
  var config1 = "chrome-extension://" + this.Key1 + "/config.json";
  var config2 = "chrome-extension://" + this.Key2 + "/config.json";
  var that = this;
  this.GetContent(config1, function(str){
    var obj = JSON.parse(str);
    if(obj){
      that.PrivateOpen(url, target, that.Key1);
    }
    else{
      console.warn("RequireInstallExtension");
      OpenT3DPage.OpenNormalPage(url, target);
    }
  },function(){
    that.GetContent(config2, function(str){
      var obj = JSON.parse(str);
      if(obj){
        that.PrivateOpen(url, target, that.Key2);
      }
      else{
        console.warn("RequireInstallExtension");
        //OpenT3DPage.OpenNormalPage(url, target);
      }
    },function(){
      console.warn("RequireInstallT3DAndExtension");
      //OpenT3DPage.OpenNormalPage(url, target);
    })
  })
}

function PostMessageToTop(msg){
  window.top.postMessage(msg, "*");
}

//如果需要可以重写此函数
//如果想飘到T3D上边，可以创建一个iframe z-index >= 10000进行占位。
OpenT3DPage.Alert = function(str){
  if(this.ShowingAlertDlg) return;
  var root = document.createElement("div");
  root.setAttribute("id", "Alert");
  root.setAttribute("style", "width: 40%; height: 25%; position: fixed; padding: 20% 30% 55% 30%; " +
      "background-color: rgba(0, 0, 0, 0.5); left: 0px; top: 0px; z-index: 999999");
  root.addEventListener("click", function(){
      document.body.removeChild(root);
      this.ShowingAlertDlg = false;
  }.bind(this), false);
  var content = document.createElement("div");
  content.setAttribute("id", "AlertContent");
  content.setAttribute("style", "width: 40%; height: 25%; background-color: white; position: fixed; z-index: 10001; ");
  content.innerHTML =
      "<div>" +
      "<h1 style='text-align: center;'>" + this.GetLan("10001") + "</h1>" +
      "<h2 style='padding: 10px; text-align: center;'>" + str + "</h2>" +
      "<input id='OK' type='button' value='" + this.GetLan("10002") + "' " +
          "style='width:50%; height: 30px; left:25%; border:0px solid black; background-color: skyblue; margin-left: 25%;'/>" +
      "</div>";
  content.addEventListener("click", function(){
      event.stopPropagation();
  }, false);
  var that = this;
  var keydownFun = function(){
    if((event.keyCode==13)){
      document.body.removeChild(root);
      that.ShowingAlertDlg = false;
    }
    document.removeEventListener("keyup", keydownFun, false);
  }
  document.addEventListener("keyup", keydownFun, false);
  content.getElementsByTagName("input")[0].addEventListener("click", function(){
      document.body.removeChild(root);
      this.ShowingAlertDlg = false;
  }.bind(this), false);
  var placeHolder = document.createElement("iframe");
  placeHolder.setAttribute("id", "AlertPlaceHolder");
  placeHolder.setAttribute("style", "width: 40%; height: 25%; position: fixed; padding: 0% 0% 0% 0%; " +
      "background-color: #ccc; border: none; z-index: 10000")
  root.appendChild(placeHolder);
  root.appendChild(content);
  document.body.appendChild(root);
  this.ShowingAlertDlg = true;
}
