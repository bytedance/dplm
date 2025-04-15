document.addEventListener("DOMContentLoaded", function () {
  // Get all carousel video elements
  var carouselVideos = document.querySelectorAll(".carousel-video");

  var options = {
    root: null, // Use the viewport as the root element
    rootMargin: "0px", // No margin applied to the root
    threshold: 0.2, // Percentage of the video visible in the viewport
  };

  // Create a new Intersection Observer
  var observer = new IntersectionObserver(function (entries, observer) {
    entries.forEach(function (entry) {
      if (entry.isIntersecting) {
        var carouselVideos = document.querySelectorAll(".carousel-video");
        console.log(carouselVideos)
         carouselVideos.forEach(function (video) {
          video.setAttribute("src", video.getAttribute("data-src"));
          video.setAttribute("class", "video");
          video.autoplay = true;
        });

        // Stop observing the video once it becomes visible
        observer.unobserve(entry.target);
      }
    });
  }, options);

  // Observe each carousel video element
  carouselVideos.forEach(function (video) {
    observer.observe(video);
  });
});