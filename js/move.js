// 动态创建iframe
function createIframe(src) {
    //在document中创建iframe
    const iframe = document.createElement("iframe");
    const box = document.createElement("div");
    const bt = document.createElement("button");
    bt.id = 'videoBt';
    bt.innerHTML = '关闭';
    box.id = 'iframeBox';
    iframe.id = 'iframeId';
    //设置iframe的样式
    box.style.width = '900px';
    box.style.height = '550px';

    box.style.position = 'absolute';
    box.style.top = '50px';
    box.style.left = '50px';
    box.style.background = '#fff';
    box.style.borderTop = '46px solid #2196f3';

    iframe.style.width = '100%';
    iframe.style.height = '100%';
    iframe.style.padding = '0';
    iframe.style.overflow = 'hidden';
    iframe.style.border = 'none';

    iframe.src = src;
    //把iframe加载到dom下面
    box.appendChild(iframe);
    box.appendChild(bt);
    document.body.appendChild(box);
    $('#videoBt').css({
        position: 'absolute',
        right: '10px',
        top: '-38px',
        background: '#2196f3',
        border: 'none',
        cursor: 'pointer',
        color: '#fff',
        padding: '5px 15px',
        border: '4px'
    })
    //绑定关闭事件
    $('#videoBt').click(function () {
        document.body.removeChild($('#iframeBox')[0])
    })

    // 给div 注册事件
    // document.getElementById('iframeBox').addEventListener("onmousedown",function (ev){
    //     const body = document.body;
    //     var oEvent = ev || event;       //判断浏览器兼容
    //     disX = oEvent.clientX - body.offsetLeft;    //鼠标横坐标点到div的offsetLeft距离
    //     disY = oEvent.clientY - body.offsetTop;     //鼠标纵坐标点到div的offsetTop距离
    // });
    initMove();
}


function  initMove (){
    const dv = document.getElementById('iframeBox');
    let x = 0;
    let y = 0;
    let l = 0;
    let t = 0;
    let disX = 0;
    let disY = 0;
    let isDown = false;
    //鼠标按下事件
    dv.onmousedown = function(e) {
        //获取x坐标和y坐标
        x = e.clientX;
        y = e.clientY;

        //获取左部和顶部的偏移量
        l = dv.offsetLeft;
        t = dv.offsetTop;
        //开关打开
        isDown = true;
        //设置样式
        dv.style.cursor = 'move';
    }
    //鼠标移动
    window.onmousemove = function(e) {
        if (isDown == false) {
            return;
        }
        //获取x和y
        const nx = e.clientX;
        const ny = e.clientY;
        //计算移动后的左偏移量和顶部的偏移量
        const nl = nx - (x - l);
        const nt = ny - (y - t);

        dv.style.left = nl + 'px';
        dv.style.top = nt + 'px';
    }
//鼠标抬起事件
    window.onmouseup = function() {
        //开关关闭
        isDown = false;
        dv.style.cursor = 'default';
    }
    dv.onmouseup = function() {
        //开关关闭
        isDown = false;
        dv.style.cursor = 'default';
    }
}
