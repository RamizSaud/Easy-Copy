import React, { useState } from "react";
import { ScrollView, View, Text, Image, StyleSheet } from "react-native";
import * as ImagePicker from "expo-image-picker";
import * as Clipboard from "expo-clipboard";

const App = () => {
  const [imageUri, setImageUri] = useState(null);
  const [text, setText] = useState(null);

  const pickImage = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      setImageUri(result.assets[0].uri);

      const formData = new FormData();
      formData.append("image", {
        uri: result.assets[0].uri,
        type: "image/jpeg",
        name: "my-image.jpg",
      });

      try {
        const response = await fetch("http://192.168.137.233:5000/upload", {
          method: "POST",
          body: formData,
          headers: {
            "Content-Type": "multipart/form-data",
          },
        });

        const result = await response.text();
        console.log(result);
      } catch (error) {
        console.error("Error uploading image:", error);
      }
    }
  };

  const copyToClipboard = async () => {
    await Clipboard.setStringAsync(text);
  };

  async function sendGetRequest() {
    try {
      const response = await fetch("http://192.168.137.233:5000", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json();
      setText(data.text);
      return data;
    } catch (error) {
      console.error("Error:", error);
      throw error;
    }
  }

  return (
    <View style={styles.main}>
      <Image source={require("./assets/logo.png")} style={styles.image} />
      <View style={styles.container}>
        <Text style={styles.title}>Image to Text</Text>
        <View style={styles.btn_container}>
          <Text style={styles.btn} onPress={pickImage}>
            Upload Image
          </Text>
        </View>
        <View style={styles.image_container}>
          {imageUri && (
            <>
              <Image source={{ uri: imageUri }} style={styles.uploaded_image} />
              <View style={styles.convert_container}>
                <Text style={styles.convert} onPress={sendGetRequest}>
                  Convert to Text
                </Text>
              </View>
            </>
          )}
        </View>
        {text && (
          <>
            <ScrollView style={styles.scrollView}>
              <View style={styles.text_container}>
                <Text style={styles.generated_text}>{text}</Text>
              </View>
            </ScrollView>
            <View style={styles.copy_btn}>
              <Text onPress={copyToClipboard} style={styles.copy}>
                Copy Text
              </Text>
            </View>
          </>
        )}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  main: {
    flex: 1,
    backgroundColor: "#E0F7FA",
    alignItems: "center",
  },
  image: {
    width: 150,
    height: 150,
    marginTop: 10,
  },
  container: {
    justifyContent: "space-between",
    alignItems: "center",
    height: "15%",
  },
  title: {
    fontSize: 42,
    marginBottom: 5,
    color: "#0277BD",
    fontWeight: "bold",
    marginTop: -20,
  },
  btn_container: {
    width: 250,
    color: "red",
    backgroundColor: "#0277BD",
    borderRadius: 10,
  },
  btn: {
    color: "#fff",
    padding: 15,
    textAlign: "center",
    fontSize: 20,
  },
  uploaded_image: {
    width: 250,
    height: 250,
    borderRadius: 10,
  },
  image_container: {
    marginTop: 10,
    width: "100%",
  },
  text_container: {
    margin: 20,
  },
  generated_text: {
    fontSize: 20,
  },
  scrollView: {
    marginTop: 10,
    width: 320,
    height: 400,
    paddingBottom: 120,
    backgroundColor: "white",
    marginBottom: 15,
  },
  copy_btn: {
    width: 120,
    backgroundColor: "#0277BD",
    padding: 10,
    borderRadius: 10,
  },
  copy: {
    color: "white",
    textAlign: "center",
    fontSize: 15,
  },
  convert: {
    backgroundColor: "#0277BD",
    padding: 10,
    textAlign: "center",
    marginTop: 10,
    color: "white",
    width: 150,
    borderRadius: 10,
  },
  convert_container: {
    alignItems: "center",
  },
});

export default App;
