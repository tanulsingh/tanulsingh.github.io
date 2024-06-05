document.addEventListener("DOMContentLoaded", function () {
    const toggleSwitch = document.querySelector("#theme-toggle");
    const currentTheme = localStorage.getItem("theme");

    if (currentTheme) {
        document.body.classList.add(currentTheme);
    }

    toggleSwitch.addEventListener("change", function () {
        if (this.checked) {
            document.body.classList.add("dark-theme");
            localStorage.setItem("theme", "dark-theme");
        } else {
            document.body.classList.remove("dark-theme");
            localStorage.removeItem("theme");
        }
    });
});