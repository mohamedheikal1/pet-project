document.getElementById("upload-form").addEventListener("submit", function(e){
     e.preventDefault();


     let formData = new FormData(this);

    fetch("/upload", {
         method: "POST",
         body: formData
     
        })

        .then(response => response.json())
        .then(data => {
            document.getElementById("result-container").innerText = 'Result:' + data.result;
        })
        .catch(error => {
          console.error(error);
        });
     });

